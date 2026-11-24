"""The face dock consumes exactly the original Live Link playback sample."""
import numpy as np
from .face_data import CHANNELS
from .face_extract import ARKIT

PREVIEW_ORDER = np.asarray([CHANNELS.index(name) for name in ARKIT])


def preview_sample(window, *, visible=True):
    zero = np.zeros(len(ARKIT), dtype=np.float32)
    frame = int(window.motion_view.index)
    if not visible or window._pear_active or window.busy:
        return zero, frame, 'Track Face', 'Preview inactive', None
    person = window.current_person() if window.people_state else None
    ident = person['id'] if person else 1
    label = person.get('name', 'Person ' + str(ident)) if person else 'Person 1'
    if window.people_state and (person is None or window.scene_data is None):
        return zero, frame, label, 'Select a person with prepared motion', None
    try:
        sample = window.stream_face_at_playhead(ident)
        if sample is None:
            return zero, frame, label, 'No Track Face data at this frame', None
        values = np.asarray(sample['values'], dtype=np.float32)
        if values.shape != (len(CHANNELS),) or not np.isfinite(values).all():
            raise ValueError('Invalid Track Face expression values')
        archive = window.face_playback_archive(window.face_project())
        neutral = archive.neutral_frames.get(ident) if archive is not None else None
        return (values[PREVIEW_ORDER].copy(), int(sample['source_frame']), label,
                'Ready' if sample['tracked'] else 'Held expression', neutral)
    except (OSError, ValueError, KeyError, IndexError, TypeError) as error:
        return zero, frame, label, 'Track Face: ' + str(error), None
