/* Durable local queue for a survey recorded with no server in reach.
 *
 * When the phone runs the model itself there is no reason to be in touch with
 * the server during a drive, and in a moving car there very often is no way
 * to be. Results are therefore written to IndexedDB - which survives the tab
 * being backgrounded, the browser being killed and the phone running out of
 * battery - and posted in one batch afterwards.
 *
 * Every queued detection carries a client-minted id. The sync endpoint treats
 * that id as unique, so a batch interrupted halfway and retried cannot record
 * the same pothole twice.
 */

const DB_NAME = 'potholesense';
const DB_VERSION = 1;

function open() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('detections'))
        db.createObjectStore('detections', { keyPath: 'client_id' });
      if (!db.objectStoreNames.contains('track'))
        db.createObjectStore('track', { autoIncrement: true });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

function tx(db, store, mode, fn) {
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const result = fn(t.objectStore(store));
    t.oncomplete = () => resolve(result && result.result !== undefined
      ? result.result : result);
    t.onerror = () => reject(t.error);
  });
}

export const newId = () => (crypto.randomUUID ? crypto.randomUUID()
  : `${Date.now()}-${Math.random().toString(36).slice(2)}`);

export async function queueDetection(record) {
  const db = await open();
  return tx(db, 'detections', 'readwrite', s => s.put(record));
}

export async function queueTrackPoint(point) {
  const db = await open();
  return tx(db, 'track', 'readwrite', s => s.add(point));
}

export async function pending() {
  const db = await open();
  const detections = await tx(db, 'detections', 'readonly', s => s.getAll());
  const track = await tx(db, 'track', 'readonly', s => s.getAll());
  return { detections, track };
}

export async function clear() {
  const db = await open();
  await tx(db, 'detections', 'readwrite', s => s.clear());
  await tx(db, 'track', 'readwrite', s => s.clear());
}

/* Post everything held locally. Only clears on a confirmed success, so a
 * failed sync leaves the survey intact for the next attempt. */
export async function flush(sessionId) {
  const { detections, track } = await pending();
  if (!detections.length && !track.length) return { accepted: 0, pending: 0 };

  const res = await fetch('/api/sync', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId, detections, track }),
  });
  if (!res.ok) throw new Error(`sync rejected (${res.status})`);
  const result = await res.json();
  await clear();
  return { ...result, pending: 0 };
}
