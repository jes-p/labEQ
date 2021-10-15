"Additional utils for labEQ"

def parse_eid(event_id):
    """Parse an event_str or integer event_num input into the event_str"""
    if isinstance(event_id,int):
        event_str = f'event_{event_id:0>3d}'
    else:
        event_str = event_id
    return event_str

def get_event(ds, event_id):
    """Get event_str, tag, and trace_num for an event in the dataset"""
    event_str = parse_eid(event_id)
    edict = ds.auxiliary_data.LabEvents[event_str].parameters
    tag = edict['tag']
    tnum = edict['trace_num']
    return event_str, tag, tnum