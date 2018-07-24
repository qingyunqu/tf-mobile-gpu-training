from functools import wraps

__author__ = 'afpro'
__email__ = 'admin@afpro.net'

__all__ = [
    'isolator',
]


class _Isolator:
    def __init__(self, fget):
        assert fget is not None
        self._fget = fget
        self._fset = None
        self._fdel = None

    def __get__(self, instance, owner):
        return self._fget(instance)

    def __set__(self, instance, value):
        if self._fset is None:
            raise NotImplementedError('fset is None')
        self._fset(instance, value)

    def __delete__(self, instance):
        if self._fdel is None:
            raise NotImplementedError('fdel is None')
        self._fdel(instance)

    def __call__(self, fn):
        @wraps(fn)
        def inner(obj, *args, **kwargs):
            session = self._fget(obj)
            with session.graph.as_default(), session.as_default():
                return fn(obj, *args, **kwargs)

        return inner

    def setter(self, fn):
        self._fset = fn
        return fn

    def deleter(self, fn):
        self._fdel = fn
        return fn


def isolator(fget):
    """
    class Sample:
        def __init__(self):
            self._sess1 = tf.Session(graph=tf.Graph())
            self._sess2 = tf.Session(graph=tf.Graph())

        @isolator
        def session1(self):
            return self._sess1

        @isolator
        def session2(self):
            return self._sess2

        @session1
        def in_sess1(self, s):
            # now tf.get_default_session() is self._sess1
            pass

        @session2
        def in_sess2(self, s):
            # now tf.get_default_session() is self._sess2
            print(s)

        def as_property(self):
            s = self.session1 # can be used as property

        @session1.setter
        def set_session1(self, session):
            # also support deleter
            self._sess1 = session
    """
    return _Isolator(fget)
