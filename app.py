"""Streamlit token browser and persistent review panel."""

from __future__ import annotations

import io
import json
import wave

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from microevolution.project import Project, ProjectError, ReviewStore, export_zip, merged_tokens


st.set_page_config(page_title="Vowel microevolution", layout="wide")
st.title("Longitudinal vowel review")
st.caption("Local, provenance-preserving pilot · points are tokens, not independent sessions")

manifest_text = st.sidebar.text_input("Project manifest", "demo/project.json")
try:
    project = Project.load(manifest_text)
except (OSError, KeyError, ValueError, ProjectError, json.JSONDecodeError) as exc:
    st.error(f"Could not load project: {exc}")
    st.stop()

reviews = ReviewStore(project.root / "reviews.sqlite3")
rows = merged_tokens(project, reviews)
recordings = {r["recording_id"]: r for r in project.recordings}
for row in rows:
    rec = recordings[row["recording_id"]]
    row["speaker_id"] = rec["speaker_id"]
    row["effective_date"] = rec["recording_date"] or rec["upload_date"] or None
    row["date_display"] = (rec["recording_date"] and f"{rec['recording_date']} (recorded)") or \
        (rec["upload_date"] and f"{rec['upload_date']} (upload proxy)") or "Unknown"

speaker = st.sidebar.selectbox("Speaker", sorted({r["speaker_id"] for r in rows}))
phone = st.sidebar.selectbox("Target vowel", sorted({r["phone_label"] for r in rows if r["speaker_id"] == speaker}))
origin = st.sidebar.selectbox("Data origin", ["observed", "synthetic"],
                              help="Synthetic and observed data can never be combined here.")
statuses = st.sidebar.multiselect("Review status", ["pending", "accepted", "rejected"],
                                  default=["pending", "accepted"])
filtered = [r for r in rows if r["speaker_id"] == speaker and r["phone_label"] == phone
            and r["data_origin"] == origin and r["review_status"] in statuses]
dated = [r for r in filtered if r["effective_date"]]
unknown_count = len(filtered) - len(dated)
if unknown_count:
    st.warning(f"{unknown_count} selected token(s) have unknown dates and are excluded from timelines—not assigned a synthetic date.")

df = pd.DataFrame(dated)
for col in ("f1_hz", "f2_hz"):
    if col in df:
        df[col] = pd.to_numeric(df[col], errors="coerce")
if "effective_date" in df:
    df["effective_date"] = pd.to_datetime(df["effective_date"])

c1, c2, c3 = st.columns(3)
c1.metric("Tokens displayed", len(filtered))
c2.metric("Independent recordings", len({r["recording_id"] for r in filtered}))
c3.metric("Accepted", sum(r["review_status"] == "accepted" for r in filtered))

left, right = st.columns(2)
with left:
    st.subheader("Timeline")
    if df.empty:
        st.info("No dated measured tokens in this selection.")
    else:
        long = df.melt(id_vars=["token_id", "effective_date", "word", "recording_id"],
                       value_vars=["f1_hz", "f2_hz"], var_name="measure", value_name="Hz")
        fig = px.scatter(long, x="effective_date", y="Hz", color="measure", symbol="word",
                         hover_data=["token_id", "recording_id"], trendline=None)
        st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Vowel space")
    measured = df.dropna(subset=["f1_hz", "f2_hz"]) if not df.empty else df
    if measured.empty:
        st.info("No valid F1/F2 pairs in this selection.")
    else:
        fig = px.scatter(measured, x="f2_hz", y="f1_hz", color="word",
                         hover_data=["token_id", "recording_id", "date_display"])
        fig.update_xaxes(autorange="reversed", title="F2 (Hz; decreases →)")
        fig.update_yaxes(autorange="reversed", title="F1 (Hz; increases ↓)")
        st.plotly_chart(fig, use_container_width=True)

st.subheader("Original-audio review")
if not filtered:
    st.info("No tokens match the filters.")
    st.stop()
token_id = st.selectbox("Token", [r["token_id"] for r in filtered])
token = next(r for r in filtered if r["token_id"] == token_id)
rec = recordings[token["recording_id"]]
st.write(f"**{token['word']}** /{token['phone_label']}/ · {token['date_display']} · "
         f"original interval {float(token['original_start_s']):.3f}–{float(token['original_end_s']):.3f} s · "
         f"recording `{rec['recording_id']}`")
context = project.context_wav(token, padding_s=0.5)
st.audio(context, format="audio/wav")

with wave.open(io.BytesIO(context), "rb") as wav:
    rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    signal = np.frombuffer(wav.readframes(wav.getnframes()), dtype=dtype)
    if channels > 1:
        signal = signal.reshape(-1, channels).mean(axis=1)
time = np.arange(len(signal)) / rate
wave_fig = go.Figure(go.Scatter(x=time, y=signal, mode="lines", line={"width": 1}))
relative_start = min(0.5, float(token["original_start_s"]))
wave_fig.add_vrect(x0=relative_start, x1=relative_start + float(token["original_end_s"]) - float(token["original_start_s"]),
                   fillcolor="gold", opacity=0.25, line_width=0)
wave_fig.update_layout(height=230, xaxis_title="Seconds in playback excerpt", yaxis_title="Amplitude")
st.plotly_chart(wave_fig, use_container_width=True)
window = min(512, max(64, 2 ** int(np.log2(max(64, len(signal) // 20)))))
if len(signal) >= window:
    starts = np.arange(0, len(signal) - window + 1, max(1, window // 4))
    spectra = np.stack([np.abs(np.fft.rfft(signal[i:i + window] * np.hanning(window))) for i in starts])
    frequencies = np.fft.rfftfreq(window, 1 / rate)
    keep = frequencies <= 6000
    spectrogram = go.Figure(go.Heatmap(
        x=starts / rate, y=frequencies[keep],
        z=20 * np.log10(spectra[:, keep].T + 1), colorscale="Magma", colorbar_title="dB",
    ))
    spectrogram.add_vline(x=relative_start, line_color="cyan")
    spectrogram.add_vline(x=relative_start + float(token["original_end_s"]) - float(token["original_start_s"]),
                          line_color="cyan")
    spectrogram.update_layout(height=300, xaxis_title="Seconds in playback excerpt",
                              yaxis_title="Frequency (Hz)", title="Spectrogram and phone boundaries")
    st.plotly_chart(spectrogram, use_container_width=True)
try:
    trajectory = json.loads(token["trajectory"] or "[]")
except json.JSONDecodeError:
    trajectory = []
if trajectory:
    track = pd.DataFrame(trajectory).melt(id_vars="position", value_vars=["f1_hz", "f2_hz"],
                                          var_name="track", value_name="Hz")
    st.plotly_chart(px.line(track, x="position", y="Hz", color="track", markers=True,
                            title="Praat/Burg formant track (relative vowel position)"), use_container_width=True)

with st.form("review"):
    status = st.radio("Decision", ["pending", "accepted", "rejected"],
                      index=["pending", "accepted", "rejected"].index(token["review_status"]), horizontal=True)
    reason = st.text_input("Reason / correction note", token.get("exclusion_reason", ""))
    if st.form_submit_button("Save review"):
        reviews.set(token_id, status, reason)
        st.success("Saved persistently. Reloading counts and aggregates…")
        st.rerun()

st.download_button("Export displayed selection", export_zip(project, reviews, [r["token_id"] for r in filtered]),
                   file_name=f"{speaker}-{phone}-{origin}.zip", mime="application/zip")
