"""POSIX evaluation-only wall-clock guard; product budgets remain cooperative."""
from contextlib import contextmanager
import signal
from threading import current_thread, main_thread


class EvaluationDeadlineExceeded(RuntimeError):
    """One registered evaluation attempt exhausted its hard wall-clock limit.

    Deliberately not an OSError/TimeoutError: HTTP transports may translate
    those into network errors, obscuring the registered evaluation deadline.
    """


@contextmanager
def evaluation_deadline(seconds: float):
    """Interrupt a synchronous model/tool call while allowing its trace to finalize.

    This guard is for the main thread of the dedicated evaluation worker. It
    cannot guarantee cancellation of work already accepted by a remote server.
    Native code may defer Python signal handling; actual elapsed time is saved.
    """
    if seconds <= 0 or current_thread() is not main_thread():
        raise ValueError('A positive deadline on the main evaluation thread is required.')
    if not hasattr(signal, 'setitimer'):
        raise RuntimeError('The evaluation hard deadline requires POSIX setitimer.')
    if signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        raise RuntimeError('Refusing to replace an existing alarm timer.')
    previous = signal.getsignal(signal.SIGALRM)

    def expired(signum, frame):
        raise EvaluationDeadlineExceeded(f'Evaluation hard deadline of {seconds:g} seconds exceeded.')

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
