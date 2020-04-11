# Introduction to the evaluation interface
The simultaneous translation models from the shared task participants are evaluated under a server-client protocol. 
The participants are required to plug in their own model API in the protocol, and submit a Docker file.
The server provides information needed by the client and evaluates latency and quality, while the client sends the translation. 
We use the Fairseq toolkit as an example but the evaluation process can be applied in an arbitary framework.

## Server
The server code is provided and can be set up locally for development purposes. For example, to evaluate a text simultaneous test set,

```shell
python $user_dir/eval/server.py \
    --tokenizer 13a \
    --src-file $src \
    --tgt-file $tgt \
    --scorer-type {text, speech} \
    --output $result_dir/eval \
    --port 12321
```
The `--score-type` can be either `text` or `speech` to evaluation different tasks.

For text models, the `$src` and `$tgt` are the raw source and target text.

For speech models, please first follow the Data Preparation [here](baseline.md) except for the binarization step. After preparation, Only `$tgt` is need, which is the json file in the data directory (for example, dev.json)

The state sent to the client by the server has the following format
```json
{
  'sent_id': Int,
  'segment_id': Int,
  'segment': String or speech utterance
}
```
For text, the segment is a detokenized word, while for speech, it is a list of numbers.

## Client
The client will load model, get source tokens from server and send translated target tokens to server. 
We have an out-of-box implementation of client which could save time from configuring the communication between the server and client.
It's in fairseq repository but it can be applied to an arbitary toolkit.
If you plan to set up your client from scratch, please refer to [Evaluation Server API](server_api.md)


### Agent 
The core of the client module is the [agent](../eval/agents/agent.py). 
One can build a customized agent from the abstract class of the agent, shown as follows.

```python
class Agent(object):
    "an agent needs to follow this pattern"
    def __init__(self, *args, **kwargs):
        ...

    def init_states(self):
        # Initializing states
        ...

    def update_states(self, states, new_state):
        # Update states with given new state from server
        # format of the new_state
        # {
        #   'sent_id': Int,
        #   'segment_id': Int,
        #   'segment': String or speech utterance
        # }
        ...

    def finish_eval(self, states, new_state):
        # Check if evaluation is finished
        # Return True if finished False not
        ...
    
    def policy(self, states):
        # Provide a action given current states
        # The action can only be either
        # Speech {key: "GET", value: {"segment_size": segment_size}}
        # Text {key: "GET", value: None}
        # or
        # {key: "SEND", value: W}
        ...

    def reset(self):
        # Reset agent
        ...
        
 def _decode_one(self, session, sent_id):
        """
        The evaluation process for one sentence happens in the function, 
        you don't have to modify this function. 
        """
        action = {}
        self.reset()
        states = self.init_states()
        while action.get('value', None) != DEFAULT_EOS:
            # take an action
            action = self.policy(states)
            if action['key'] == GET:
                new_states = session.get_src(sent_id, action["value"])
                states = self.update_states(states, new_states)

            elif action['key'] == SEND:
                session.send_hypo(sent_id, action['value'])
                    states = self.init_states() 
                    self.reset()
            else:
                raise NotImplementedError

 
```
The variable **states** is the context for the model to make the decision.
You could customize your one states format
 `init_states` will be called at beginning of translating every sentnece. 
`update_states` will be called everytime a new segement was requested from server

The `policy` function returns the action you make given the current states. 
The actions can be one of the tree as follow.
|Action|Content|
|:---:|:---:|
|Request new word (Text)| ```{key: "GET", value: None}```|
|Request new utterence (Speech) | ```{key: "GET", value: {"segment_size": segment_size}}```|
|Predict word "W"| ```{key: "SEND", value: "W"}```|

Here is an example of how to develop a customized agent. 
First of all, a name needs to be registered, here is "my_agent". 
Next, override the agent functions according to the translation model. 
The functions that need to be overriden are listed in the `MyAgent` class as follow. 
The implementation should be done in this [directory](../eval/agents) in the local fairseq repository. 

```python
from . import register_agent
from . agent import Agent
@register_agent("my_agent")
class MyAgent(Agent):
    @staticmethod
    def add_args(parser):
        # add customized arguments here

    def __init__(self, *args, **kwargs):
        ...

    def init_states(self):
        ...

    def update_states(self, states, new_state):    
        ...

    def finish_eval(self, states, new_state):
        ...
    
    def policy(self, states):
        ...

    def reset(self):
        ...

```

Here are the implementations of agents for [text *wait-k* model](../eval/agents/simul_trans_text_agent.py) and [speech *wait-k* model](../eval/agents/simul_trans_speech_agent.py). You can start your agent from these implementation. For example, for a text agent with your own, you can just override the following functions 
- `SimulTransTextAgent.load_model()`: to load a customized model.
- `SimulTransTextAgent.model.decision_from_states()`:  to make read or write decision from states.
- `SimulTransTextAgent.model.predict_from_states()` to predition a target token from states.

But you might also need to change other places for tokenization.

Once the agent is implemented, assuming there is already a server listening at port 12321, to start the evaluation, 
```
python $fairseq_dir/examples/simultanesous_translation/eval/evaluate.py \
    --port 12321 \
    --agent-type my_agent_name \
    --reset-server \
    --num-threads 4 \
    --scores \
    --model-args ... # defined in MyAgent.add_args, such as model path, tokenizer etc.
```

It can be very slow to evaluate the speech model utterance by utterance. See [here](../scripts/start-multi-client.sh) for a faster implementation, which split the evaluation set into chunks.

### Quality

The quality is measured by detokenized BLEU. So make sure that the predicted words sent to server are detokenized.

### Latency
The latency metrics are 
* Average Proportion
* Average Lagging
* Differentiable Average Lagging

For text, the unit is detokenized token.
For speech, the unit is millisecond.
