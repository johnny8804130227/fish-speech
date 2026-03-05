import html
import time
from functools import partial
from typing import Any, Callable

from fish_speech.i18n import i18n
from fish_speech.utils.schema import ServeReferenceAudio, ServeTTSRequest


def inference_wrapper(
    text,
    reference_id,
    reference_audio,
    reference_text,
    max_new_tokens,
    chunk_length,
    top_p,
    repetition_penalty,
    temperature,
    seed,
    use_memory_cache,
    engine,
):
    """
    Wrapper for the inference function.
    Used in the Gradio interface.
    When chunk_length > 0, yields intermediate audio segments for real-time playback.
    """

    if reference_audio:
        references = get_reference_audio(reference_audio, reference_text)
    else:
        references = []

    use_streaming = chunk_length > 0

    req = ServeTTSRequest(
        text=text,
        reference_id=reference_id if reference_id else None,
        references=references,
        max_new_tokens=max_new_tokens,
        chunk_length=chunk_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        temperature=temperature,
        seed=int(seed) if seed else None,
        use_memory_cache=use_memory_cache,
        streaming=use_streaming,
    )

    yield None, None, "Status: starting..."

    t0 = time.monotonic()
    segment_times = []

    for result in engine.inference(req):
        match result.code:
            case "header":
                pass
            case "segment":
                segment_times.append(time.monotonic() - t0)
                stats = _format_stats(t0, segment_times, done=False)
                yield result.audio, None, stats
            case "final":
                stats = _format_stats(t0, segment_times, done=True)
                if not segment_times:
                    # non-streaming path: engine collected everything into "final"
                    yield result.audio, None, stats
                else:
                    yield None, None, stats
                return
            case "error":
                stats = _format_stats(t0, segment_times, done=True)
                yield None, build_html_error_message(i18n(result.error)), stats
                return

    stats = _format_stats(t0, segment_times, done=True)
    yield None, i18n("No audio generated"), stats


def _format_stats(
    t0: float,
    segment_times: list[float],
    done: bool,
) -> str:
    elapsed = time.monotonic() - t0
    status = "done" if done else "streaming"

    lines = [f"Status: {status}", f"Elapsed: {elapsed:.2f}s", f"Segments: {len(segment_times)}"]

    for i, t in enumerate(segment_times):
        duration = t - segment_times[i - 1] if i > 0 else t
        lines.append(f"  Seg {i + 1}: {t:.2f}s (+{duration:.2f}s)")

    return "  \n".join(lines)


def get_reference_audio(reference_audio: str, reference_text: str) -> list:
    """
    Get the reference audio bytes.
    """

    with open(reference_audio, "rb") as audio_file:
        audio_bytes = audio_file.read()

    return [ServeReferenceAudio(audio=audio_bytes, text=reference_text)]


def build_html_error_message(error: Any) -> str:

    error = error if isinstance(error, Exception) else Exception("Unknown error")

    return f"""
    <div style="color: red;
    font-weight: bold;">
        {html.escape(str(error))}
    </div>
    """


def get_inference_wrapper(engine) -> Callable:
    """
    Get the inference function with the immutable arguments.
    """

    return partial(
        inference_wrapper,
        engine=engine,
    )
