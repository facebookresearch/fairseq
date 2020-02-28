# Introduction to the evaluation interface
The simultaneous translation models from the shared task participants are evaluated under a server-client protocol. The participants are required to plug in their own model API in the protocol, and submit a Docker image.

## Server-Client Protocol
A server-client protocol will be used in evaluation. 
The server provides words or speech segments sequantially, while the client executes simultaneous translation policies and makes predictions.
Each time client make a decide on action base on the context and send corresponding request to server. 
A read action will send a "GET" request to server and a new state will be returned, while a write action send a "SEND" request to the server and update predictions and delays. 
The server can hence calculate both the latency and BLEU score of the prediction.
For example, when a *wait-k* model (k=3) translate the English sentence "Alice and Bob are good friends" to the German sentence "Alice und Bob sind gute Freunde." , the evaluation process is illustrated as follow.
|Inputs|Predctions|Action|Requests to the server| States from server |
|:---:|:---:|:---:|:---:|:---:|
|Alice | | Read |`{key: "GET", value: None}`| `{"sent_id": 0, "segment_id": 0, "segment": "Alice"}`|
|and | | Read |`{key: "GET", value: None}`| `{"sent_id": 0, "segment_id": 1, "segment": "and"}`|
|Bob | | Read |`{key: "GET", value: None}`| `{"sent_id": 0, "segment_id": 2, "segment": "Bob"}`|
| |Alice | Write |`{key: "SEND", value: "Alice"}`| 
|are | |Read |`{key: "GET", value: None}`| `{"sent_id": 0, "segment_id": 3, "segment": "are"}`|
| | und | Write |`{key: "SEND", value: "und"}`|
|good| | Read|`{key: "GET", value: None}`|`{"sent_id": 0, "segment_id": 4, "segment": "good"}`|
| | Bob| Write |`{key: "SEND", value: "Bob"}`|
|friends.| |Read |`{key: "GET", value: None}`|`{"sent_id": 0, "segment_id": 5, "segment": "friends"}`|
| | sind| Write |`{key: "SEND", value: "sind"}`|
| <\s>| | Read |`{key: "GET", value: None}`|`{"sent_id": 0, "segment_id": 6, "segment": "<\s>"}`|
| | gute| Write |`{key: "SEND", value: "gute"}`|
| | freunde.| Write |`{key: "SEND", value: "freunde"}`|
| | <\s>| Write |`{key: "SEND", value: "<\s>"}`|

.

### Server
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
The `--score-type` can be either `text` or `speech` to evaluation different task.

The state that server sent to client is has the following format
```json
{
  'sent_id': Int,
  'segment_id': Int,
  'segment': String or speech utterance
}
```
For text the segment is a detokenized word, while for speech is a list of numbers.

### Client
The client will handle the evaluation process mentioned above. It should be out-of-box as well. The client's protocol is as following table.  The segment_size the length of segment in milisecond.

|Action|Content|
|:---:|:---:|
|Request new word| ```{key: "GET", value: None}```|
|Request new utterence | ```{key: "GET", value: {"segment_size": segment_size}}```|
|Predict word "W"| ```{key: "SEND", value: "W"}```|

The core of the client module is the agent, which needs to be modified for different models accordingly. The abstract class of agent is as follow, the evaluation process for one sentence happens in the `_decode_one()` function.

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
        ...
    
    def policy(self, state: list) -> dict:
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
Here are the implementations of agents for [text *wait-k* model](../eval/agents/simul_trans_text_agent.py) and [speech *wait-k* model](../eval/agents/simul_trans_speech_agent.py).

## Evaluation
### Quality
We evaluate the detokenized BLEU using [sacreBLEU](https://github.com/mjpost/sacreBLEU) with tokenizer set to `13a`. Therefore, the agent should sent detokenized token to the server.

### Latency
The latency metrics are 
* Average Proportion
* Average Lagging
* Differentiable Average Lagging

For text, they will be evaluated on detokenized text. For speech, the will be evaluated based one millisecond
