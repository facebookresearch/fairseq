from simuleval.utils import entrypoint
from fairseq.models import streaming


@entrypoint
class S2TSystem(streaming.agents.TestTimeWaitKS2T):
    __doc__ = """
    Test time waitk speech to text pipeline
    1. offline W2V encoder
    2. test-time waitk decoder
    3. spm detokenizer
    """
