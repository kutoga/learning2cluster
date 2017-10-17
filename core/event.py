
class Event:
    def __init__(self):
        self._handlers = []
        self._handlers_nth = {}

    def add(self, handler, nth=None):
        if nth is None:
            self._handlers.append(handler)
        else:
            if nth not in self._handlers_nth:
                self._handlers_nth[nth] = []
            self._handlers_nth[nth].append(handler)

    def remove(self, handler):
        self._handlers.remove(handler)
        for nth in list(self._handlers_nth.keys()):
            handlers = self._handlers_nth[nth]
            handlers.remove(handler)
            if len(handlers) == 0:
                del self._handlers_nth[nth]

    def clear(self):
        self._handlers.clear()
        self._handlers_nth.clear()

    def fire(self, *args, nth=None):
        self._execute_handlers(self._handlers, *args)
        if nth is not None:
            for lnth in self._handlers_nth.keys():
                if nth % lnth == 0:
                    self._execute_handlers(self._handlers_nth[lnth], *args)

    def _execute_handlers(self, handlers, *args):
        for handler in handlers:
            handler(*args)
