# src/football_model/pipeline/sequences.py

# def merge_actions(seq, actions=("Pass","Carry","Pressure","Shot")):
#     seq = list(seq); i = 0
#     while i < len(seq)-1:
#         if seq[i]["state"]==seq[i+1]["state"] and seq[i]["state"] in actions:
#             seq[i]["duration"] += seq[i+1]["duration"]
#             seq.pop(i+1)
#         else:
#             i += 1
#     return seq
def merge_actions(seq, actions=("Pass","Carry","Pressure","Shot")):
    return seq

def build_sequences(
    df,
    absorbing_states=("Goal","Loss","Stoppage"),
    merge_actions_after=("Pass","Carry","Pressure","Shot"),
    debug=False
):
    sequences=[]
    grouped = df.groupby(["match_id","period"], sort=False)
    for _, g in grouped:
        g2 = g.sort_values("timestamp")
        curr, last, pend = [], None, False
        for row in g2.itertuples():
            st, dur = row.states, row.duration
            loc = getattr(row,"location_transformed", row.location)
            if pend:
                if st=="Stoppage":
                    curr.append(dict(state=st,duration=dur,location=loc))
                    sequences.append(curr); curr=[]; pend=False; last=st; continue
                if st=="Goal":
                    if curr and curr[-1]["state"]=="Loss": curr.pop()
                    curr.append(dict(state=st,duration=dur,location=loc))
                    filt=[e for e in curr if e["state"] not in ("Loss","Stoppage")]
                    sequences.append(filt); curr=[]; pend=False; last=st; continue
                sequences.append(curr); curr=[]; pend=False
            if last=="Stoppage" and st=="Loss":
                last=st; continue
            if not curr and st in absorbing_states:
                last=st; continue
            if st=="Loss":
                curr.append(dict(state=st,duration=dur,location=loc))
                pend=True; last=st; continue
            curr.append(dict(state=st,duration=dur,location=loc)); last=st
            if st in absorbing_states:
                # Correction position absorbant
                if st == "Goal":
                    shot_end = getattr(row, "shot_end_location", None)
                    if shot_end:
                        loc = eval(shot_end) if isinstance(shot_end, str) else shot_end
                        if isinstance(loc, (list, tuple)) and len(loc) > 2:
                            loc = loc[:2]
                    else:
                        loc = [120.0, 40.0]
                elif st in ("Loss", "Stoppage"):
                    # Cherche la position de fin de l'action précédente
                    prev = curr[-1] if curr else None
                    end_loc = None
                    if prev:
                        for key in ("pass_end_location", "carry_end_location", "goalkeeper_end_location", "end_location"):
                            end_loc = getattr(row, key, None)
                            if end_loc:
                                break
                        if end_loc:
                            loc = eval(end_loc) if isinstance(end_loc, str) else end_loc
                        else:
                            loc = prev["location"]
                curr.append(dict(state=st, duration=dur, location=loc))
                if st=="Goal":
                    filt=[e for e in curr if e["state"] not in ("Loss","Stoppage")]
                    sequences.append(filt)
                else:
                    sequences.append(curr)
                curr=[]; pend=False
        if curr:
            # If sequence doesn't end with an absorbing state, add a Stoppage
            if not curr[-1]["state"] in absorbing_states:
                curr.append(dict(state="Stoppage", duration=0.0, location=curr[-1]["location"]))
            sequences.append(curr)
        elif pend:
            # Handle pending sequences by adding a Stoppage
            curr.append(dict(state="Stoppage", duration=0.0, location=curr[-1]["location"]))
            sequences.append(curr)
    
    return [merge_actions(s, actions=merge_actions_after) for s in sequences]
