import gradio as gr
import os
from coach_policy import CoachPolicyAgent

# Set environment variable for stub MFA
os.environ["USE_STUB_MFA"] = "True"

def analyze_audio(audio_file):
    if audio_file is None:
        return "", "", "", "", "", ""
    
    # Initialize the coach policy agent
    agent = CoachPolicyAgent()
    
    # Analyze the audio file
    result = agent.analyze(audio_file)
    
    # Extract transcript
    transcript = result.get("transcript", "")
    
    # Extract argument feedback
    argument_feedback = result.get("argument_feedback", {})
    has_claim = argument_feedback.get("has_claim", False)
    has_reason = argument_feedback.get("has_reason", False)
    has_evidence = argument_feedback.get("has_evidence", False)
    score = argument_feedback.get("score", 0)
    summary = argument_feedback.get("summary", "")
    
    # Extract pronunciation feedback
    pronunciation_feedback = result.get("pronunciation_feedback", [])
    pronunciation_text = ""
    if pronunciation_feedback:
        pronunciation_text = "\n".join([
            f"Word: {item['word']} | Expected: {item['expected_ipa']} | Actual: {item['actual_ipa']}"
            for item in pronunciation_feedback
        ])
    
    return (
        transcript,
        has_claim,
        has_reason, 
        has_evidence,
        f"Score: {score}/10",
        summary,
        pronunciation_text
    )

# Create Gradio interface
with gr.Blocks(title="SpeakFlow AI Coach") as demo:
    gr.Markdown("# SpeakFlow AI Speech Coach")
    gr.Markdown("Record your speech and get feedback on arguments and pronunciation")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="Record Audio"
            )
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column():
            transcript_box = gr.Textbox(
                label="Transcript",
                lines=5,
                interactive=False
            )
    
    gr.Markdown("## Argument Feedback")
    with gr.Row():
        claim_check = gr.Checkbox(label="Has Claim", interactive=False)
        reason_check = gr.Checkbox(label="Has Reason", interactive=False)
        evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
    
    score_text = gr.Textbox(label="Score", interactive=False)
    summary_text = gr.Textbox(label="Summary", lines=3, interactive=False)
    
    gr.Markdown("## Pronunciation Feedback")
    pronunciation_text = gr.Textbox(
        label="Mispronounced Words (with IPA)",
        lines=5,
        interactive=False
    )
    
    analyze_btn.click(
        fn=analyze_audio,
        inputs=[audio_input],
        outputs=[
            transcript_box,
            claim_check,
            reason_check,
            evidence_check,
            score_text,
            summary_text,
            pronunciation_text
        ]
    )

if __name__ == "__main__":
    demo.launch()