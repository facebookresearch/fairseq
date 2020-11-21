# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import ArgumentParser
from dataclasses import MISSING, dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


def eval_str_list(x, x_type=float):
    if x is None:
        return None
    if isinstance(x, str):
        x = eval(x)
    try:
        return list(map(x_type, x))
    except TypeError:
        return [x_type(x)]


class StrEnum(Enum):
    def __str__(self):
        return self.value

    def __eq__(self, other: str):
        return self.value == other

    def __repr__(self):
        return self.value


def ChoiceEnum(choices: List[str]):
    """return the Enum class used to enforce list of choices"""
    return StrEnum("Choices", {k: k for k in choices})


@dataclass
class FairseqDataclass:
    """fairseq base dataclass that supported fetching attributes and metas"""

    _name: Optional[str] = None

    @staticmethod
    def name():
        return None

    def _get_all_attributes(self) -> List[str]:
        return [k for k in self.__dataclass_fields__.keys()]

    def _get_meta(
        self, attribute_name: str, meta: str, default: Optional[Any] = None
    ) -> Any:
        return self.__dataclass_fields__[attribute_name].metadata.get(meta, default)

    def _get_name(self, attribute_name: str) -> str:
        return self.__dataclass_fields__[attribute_name].name

    def _get_default(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default
            ):
                return getattr(self, attribute_name)
        return self.__dataclass_fields__[attribute_name].default

    def _get_default_factory(self, attribute_name: str) -> Any:
        if hasattr(self, attribute_name):
            if str(getattr(self, attribute_name)).startswith("${"):
                return str(getattr(self, attribute_name))
            elif str(self.__dataclass_fields__[attribute_name].default).startswith(
                "${"
            ):
                return str(self.__dataclass_fields__[attribute_name].default)
            elif (
                getattr(self, attribute_name)
                != self.__dataclass_fields__[attribute_name].default_factory()
            ):
                return getattr(self, attribute_name)
        return self.__dataclass_fields__[attribute_name].default_factory()

    def _get_type(self, attribute_name: str) -> Any:
        return self.__dataclass_fields__[attribute_name].type

    def _get_help(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "help")

    def _get_argparse_const(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_const")

    def _get_argparse_alias(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "argparse_alias")

    def _get_choices(self, attribute_name: str) -> Any:
        return self._get_meta(attribute_name, "choices")


def gen_parser_from_dataclass(
    parser: ArgumentParser,
    dataclass_instance: FairseqDataclass,
    delete_default: bool = False,
) -> None:
    """convert a dataclass instance to tailing parser arguments"""
    import re

    def argparse_name(name: str):
        if name == "data":
            # normally data is positional args
            return name
        if name == "_name":
            # private member, skip
            return None
        return "--" + name.replace("_", "-")

    def interpret_dc_type(field_type):
        if isinstance(field_type, str):
            raise RuntimeError()
        typestring = str(field_type)
        if re.match(r"(typing.|^)Union\[(.*), NoneType\]$", typestring):
            return field_type.__args__[0]
        return field_type

    def get_kwargs_from_dc(
        dataclass_instance: FairseqDataclass, k: str
    ) -> Dict[str, Any]:
        """k: dataclass attributes"""
        field_type = dataclass_instance._get_type(k)
        inter_type = interpret_dc_type(field_type)
        if isinstance(inter_type, type) and issubclass(inter_type, List):
            field_default = dataclass_instance._get_default_factory(k)
        else:
            field_default = dataclass_instance._get_default(k)

        if isinstance(inter_type, type) and issubclass(inter_type, Enum):
            field_choices = [t.value for t in list(inter_type)]
        else:
            field_choices = None

        field_help = dataclass_instance._get_help(k)
        field_const = dataclass_instance._get_argparse_const(k)
        kwargs = {}
        if isinstance(field_default, str) and field_default.startswith("${"):
            kwargs["default"] = field_default
        else:
            if field_default is MISSING:
                kwargs["required"] = True
            if field_choices is not None:
                kwargs["choices"] = field_choices
            if (isinstance(inter_type, type) and issubclass(inter_type, List)) or (
                "List" in str(inter_type)
            ):
                if "int" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, int)
                elif "float" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, float)
                elif "str" in str(inter_type):
                    kwargs["type"] = lambda x: eval_str_list(x, str)
                else:
                    raise NotImplementedError()
                if field_default is not MISSING:
                    kwargs["default"] = ",".join(map(str, field_default))
            elif (
                isinstance(inter_type, type) and issubclass(inter_type, Enum)
            ) or "Enum" in str(inter_type):
                kwargs["type"] = str
                if field_default is not MISSING:
                    if isinstance(field_default, Enum):
                        kwargs["default"] = field_default.value
                    else:
                        kwargs["default"] = field_default
            elif inter_type is bool:
                kwargs["action"] = (
                    "store_false" if field_default is True else "store_true"
                )
                kwargs["default"] = field_default
            else:
                kwargs["type"] = inter_type
                if field_default is not MISSING:
                    kwargs["default"] = field_default

        kwargs["help"] = field_help
        if field_const is not None:
            kwargs["const"] = field_const
            kwargs["nargs"] = "?"
        return kwargs

    for k in dataclass_instance._get_all_attributes():
        field_name = argparse_name(dataclass_instance._get_name(k))
        if field_name is None:
            continue

        kwargs = get_kwargs_from_dc(dataclass_instance, k)

        field_args = [field_name]
        alias = dataclass_instance._get_argparse_alias(k)
        if alias is not None:
            field_args.append(alias)

        if "default" in kwargs:
            if isinstance(kwargs["default"], str) and kwargs["default"].startswith(
                "${"
            ):
                if kwargs["help"] is None:
                    # this is a field with a name that will be added elsewhere
                    continue
                else:
                    del kwargs["default"]
            if delete_default:
                del kwargs["default"]
        try:
            parser.add_argument(*field_args, **kwargs)
        except ArgumentError:
            pass
