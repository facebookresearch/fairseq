from unittest import TestCase

from fairseq.registry import Registry

class TestRegistry(TestCase):

    def setUp(self) -> None:
        self.registry = Registry()

    def tearDown(self) -> None:
        self.registry = None

    def test_registry_fails_lookup(self):
        with self.assertRaises(ValueError) as e:
            self.registry.get("unknown key")

    def test_registry_constructs_correctly(self):
        registration_key = "foo"

        @self.registry.register(registration_key)
        class Foo:
            def __init__(self, a: int, b: str):
                self.a = a
                self.b = b

        a = 1
        b = "hello"

        foo = self.registry.get(registration_key, a=a, b=b)

        self.assertEqual(foo.a, a)
        self.assertEqual(foo.b, b)