import boto3
from botocore.exceptions import ClientError
import logging
import os

logger = logging.getLogger(__name__)

def invoke_agent(agent_id, agent_alias_id, session_id, prompt):
    try:
        client = boto3.session.Session().client(
            service_name="bedrock-agent-runtime",
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )
        response = client.invoke_agent(
            agentId=agent_id,
            agentAliasId=agent_alias_id,
            enableTrace=True,
            sessionId=session_id,
            inputText=prompt
        )

        output_text = ""
        citations = []
        trace = {}
        has_guardrail_trace = False

        for event in response.get("completion"):
            if "chunk" in event:
                chunk = event["chunk"]
                output_text += chunk["bytes"].decode()
                if "attribution" in chunk:
                    citations += chunk["attribution"]["citations"]
            if "trace" in event:
                for trace_type in ["guardrailTrace", "preProcessingTrace", "orchestrationTrace", "postProcessingTrace"]:
                    if trace_type in event["trace"]["trace"]:
                        mapped_trace_type = trace_type
                        if trace_type == "guardrailTrace":
                            if not has_guardrail_trace:
                                has_guardrail_trace = True
                                mapped_trace_type = "preGuardrailTrace"
                            else:
                                mapped_trace_type = "postGuardrailTrace"
                        if mapped_trace_type not in trace:
                            trace[mapped_trace_type] = []
                        trace[mapped_trace_type].append(event["trace"]["trace"][trace_type])

    except ClientError as e:
        raise

    return {
        "output_text": output_text,
        "citations": citations,
        "trace": trace
    }
