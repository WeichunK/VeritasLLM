import signal
import os
import logging
import sys

logger = logging.getLogger(__name__)

class TrainingController:
    """
    Handles graceful stopping and resuming of training.
    Listens for SIGINT (Ctrl+C) and checks for a stop signal file.
    """
    def __init__(self, stop_signal_file: str = "stop.signal"):
        self.stop_signal_file = stop_signal_file
        self.stop_requested = False
        self.original_sigint_handler = None
        
        # Register signal handler
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

    def _handle_sigint(self, signum, frame):
        logger.info("Received SIGINT. Requesting graceful stop...")
        self.stop_requested = True
        # Restore original handler so subsequent Ctrl+C force kills
        signal.signal(signal.SIGINT, self.original_sigint_handler)

    def should_stop(self) -> bool:
        """Checks if training should stop."""
        if self.stop_requested:
            return True
        
        if os.path.exists(self.stop_signal_file):
            logger.info(f"Stop signal file found at {self.stop_signal_file}. Requesting graceful stop...")
            self.stop_requested = True
            # Optionally remove the file
            # os.remove(self.stop_signal_file)
            return True
            
        return False
