"Additional utils for labEQ"

def parse_eid(event_id):
    """Parse an event_str or integer event_num input into the event_str"""
    if isinstance(event_id,int):
        event_str = f'event_{event_id:0>3d}'
    else:
        event_str = event_id
    return event_str


