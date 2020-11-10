## Hydra

[Hydra](https://github.com/facebookresearch/hydra) is an open-source Python framework that simplifies the development of
research and other complex applications. The key feature is the ability to dynamically create a hierarchical
configuration by composition and override it through config files and the command line. The name Hydra comes from its
ability to run multiple similar jobs - much like a Hydra with multiple heads.

## Motivation

Until recently, all components in fairseq were configured through a shared "args" namespace that was created at
application startup. Components declared their own "add_args" method to update the argparse parser, hoping that
the names would not clash with arguments from other components. While this model works for smaller applications,
as fairseq grew and became integrated into other applications, this became problematic.
In order to determine how to configure each component, one needed to a) examine what args were added by this component, and
b) read the code to figure out what shared arguments it is using that were added in other places. Reproducing
models involved sharing commands that often contained dozens of command line switches.

The model described above is still supported by fairseq for backward compatibility, but will be deprecated some time
in the future.

New components in fairseq should now create a dataclass that encapsulates all parameters required to configure this
component. The dataclass is registered along with the component, and fairseq takes care of constructing and
providing this configuration object to the component's constructor. Note that sharing parameters can optionally
still work, but one has to explicitly point to the "source of truth" (see inheritance example below).
These changes make components in fairseq
more independent and re-usable by other applications: all that is needed to create a component is to initialize its
dataclass and overwrite some of the defaults.

While configuring fairseq through command line (using either the legacy argparse based or the new Hydra based entry points) is still
fully supported, you can now take advantage of configuring fairseq completely or piece-by-piece through
hierarchical YAML configuration files. These files can also be shipped as examples that others can use to run
an identically configured job.

Additionally, Hydra has a rich and growing
[library of plugins](https://github.com/facebookresearch/hydra/tree/master/plugins) that provide functionality such as
hyperparameter sweeping (including using bayesian optimization through the [Ax](https://github.com/facebook/Ax) library),
job launching across various platforms, and more.

## Creating or migrating components

In general, each new (or updated) component should provide a companion [dataclass](https://www.python.org/dev/peps/pep-0557/). These dataclass are typically located in the same
file as the component and are passed as arguments to the register_*() functions. Top-level configs that should be
present in every fairseq application are placed in the [global](fairseq/dataclass/configs.py) config file and added
to the FairseqConfig object.

Each dataclass is a plain-old-data object, similar to a NamedTuple. These classes are decorated with a @dataclass
decorator, and typically inherit from `FairseqDataclass` (which adds some functionality for backward compatibility).
Each field must have a type, and generally has metadata (such as a help string) and a default value. Only primitive types or other config objects are allowed as
data types for each field.

 Example:


``` python
from dataclasses import dataclass, field
from fairseq.dataclass import FairseqDataclass

@dataclass
class InteractiveConfig(FairseqDataclass):
    buffer_size: int = field(
        default=0,
        metadata={
            "help": "read this many sentences into a buffer before processing them"
        },
    )
    input: str = field(
        default="-",
        metadata={"help": "file to read from; use - for stdin"},
    )
```

### Inherting values

Some components require sharing a value. For example, a learning rate scheduler and an optimizer may both need to
know the initial learning rate value. One can declare a field that, by default, will
inherit its value from another config node in the same hierarchy:

``` python
@dataclass
FairseqAdamConfig(FairseqDataclass):
    ...
    lr: List[float] = II("optimization.lr")
    ...
```

`II("optimization.lr")` is syntactic sugar for `"${optimization.lr}"` , which is the value one can use in a YAML config file or through
command line to achieve the same effect. Note that this assumes that there is an "optimization" config object
in the root config and it has a field called "lr".

### Tasks and Models

Creating Tasks and Models works same as before, except that legacy implementations now inherit from Legacy* base classes,
while new components inherit from FairseqTask and FairseqModel and provide a dataclass to the register_*() functions.

Task example:

``` python
@dataclass
class LanguageModelingConfig(FairseqDataclass):
    data: Optional[str] = field(
        default=None, metadata={"help": "path to data directory"}
    )
    ...

@register_task("language_modeling", dataclass=LanguageModelingConfig)
class LanguageModelingTask(LegacyFairseqTask):
    ...
    @classmethod
    def setup_task(cls, cfg: LanguageModelingConfig):
        ...
```

Model example:

``` python
@dataclass
class TransformerLanguageModelConfig(FairseqDataclass):
    activation_fn: ChoiceEnum(utils.get_available_activation_fns()) = field(
        default="relu", metadata={"help": "activation function to use"}
    )
    dropout: float = field(default=0.1, metadata={"help": "dropout probability"})
    ...

@register_model("transformer_lm", dataclass=TransformerLanguageModelConfig)
class TransformerLanguageModel(FairseqLanguageModel):
    ...
    @classmethod
    def build_model(cls, cfg: TransformerLanguageModelConfig, task: FairseqTask):
        ...
```

### Other components

Other components work as before, but they now take their configuration dataclass as the only constructor argument:

``` python
@dataclass
class MosesTokenizerConfig(FairseqDataclass):
    source_lang: str = field(default="en", metadata={"help": "source language"})
    ...

@register_tokenizer("moses", dataclass=MosesTokenizerConfig)
class MosesTokenizer(object):
    def __init__(self, cfg: MosesTokenizerConfig):
        ...
```

Note that if you are adding a new registry for a new set of components, you need to add it to the FairseqConfig object in
fairseq/dataclass/configs.py:

``` python
@dataclass
class FairseqConfig(object):
    ...
    my_new_registry: Any = None
```

## Training with hydra_train.py

To fully take advantage of configuration flexibility offered by Hydra, you may want to train new models using the
hydra_train.py entry point located in the fairseq_cli directory. Legacy CLI tools such as train.py,
will remain supported for the foreseeable future but will be deprecated eventually.

On startup, Hydra will create a configuration object that contains a hierarchy of all the necessary dataclasses
populated with their default values in the code. The default values are overwritten by values found in YAML files in
fairseq/config directory (which currently just set default task, optimizer, etc) and then further overwritten by values
provided through command line arguments. 

Some of the most common use cases are shown below:

### 1. Overwrite default values through command line:

```shell script
python fairseq_cli/hydra_train.py distributed_training.distributed_world_size=1 dataset.batch_size=2 task.data=data-bin \
model=transformer_lm/transformer_lm_gpt task=language_modeling optimization.max_update=5000

```

Note that along with explicitly providing values for parameters such as dataset.batch_size, this also tells Hydra to overlay configuration found in `fairseq/config/model/transformer_lm/transformer_lm_gpt.yaml`
over the default values in the dataclass. If you want to train a model without specifying a particular architecture
you can simply specify model=transformer_lm. This only works for migrated tasks and models.

### 2. Replace bundled configs with an external config:

```shell script
python fairseq_cli/hydra_train.py --config-path /path/to/external/configs --config-name wiki103
```

where /path/to/external/configs/wiki103.yaml contains:

``` yaml
# @package _group_

model:
  _name: transformer_lm
distributed_training:
  distributed_world_size: 1
dataset:
  batch_size: 2
task:
  _name: language_modeling
  data: /path/to/data
  add_bos_token: false
  max_target_positions: 1024
optimization:
  max_update: 50000
  lr: [ 0.25 ]
criterion: cross_entropy
optimizer: adam
lr_scheduler:
  _name: cosine
```

Note that here bundled configs from `fairseq/config` directory are not used, however the defaults from each dataclass will still be used (unless overwritten by your external config). 

Additionally you can choose to break up your configs by creating a directory structure in the same location as your main config file, with the names of the top-level fields
(such as "model", "dataset", etc), and placing config files with meaningful names that would populate that specific section of your
top-level config file (for example, you might have model/small_transformer_lm.yaml, model/big_transformer_lm.yaml, etc). You can then specify the correct configuration via command line, defaults in the main config, or even launch all of them as a sweep (see Hydra documentation on how to do this).

### 3. Add an external config directory to Hydra search path:

This allows combining default configuration (including using any bundled config files), while specifying your own config files for some parts of the configuration.

```shell script
python fairseq_cli/hydra_train.py distributed_training.distributed_world_size=1 dataset.batch_size=2 \
task.data=/path/to/data/ model=transformer_lm/2_layers task=language_modeling optimization.max_update=5000 \
--config-dir /path/to/external/configs

```

where /path/to/external/configs has the following structure:
```
.
+-- model
|   +-- transformer_lm
|   |   +-- 2_layers.yaml
```

and 2_layers.yaml contains a copy of transformer_lm_gpt.yaml but with decoder_layers set to 2. You can add
other configs to configure other components as well.
