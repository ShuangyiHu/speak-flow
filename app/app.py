import gradio as gr
import asyncio
import os
from response_generator import ResponseGenerator, CoachingAction, CoachingStrategy, ResponseRequest

# Set environment variable for demo
os.environ["USE_STUB_MFA"] = "True"

class SpeakFlowUI:
    def __init__(self):
        self.response_generator = ResponseGenerator()
        self.turn_count = 0
        self.prior_responses = []
        
    def analyze_audio(self, audio_file, topic, user_position):
        """Analyze audio input and return all feedback components"""
        if audio_file is None:
            return "", False, False, False, 0, "", "", ""
        
        # Mock transcript for demo
        transcript = "I think renewable energy is the key to solving climate change because it reduces carbon emissions."
        
        # Mock argument analysis
        has_claim = True
        has_reason = True  
        has_evidence = False
        argument_score = 75
        argument_summary = "Strong claim with clear reasoning, but needs supporting evidence."
        
        # Mock pronunciation feedback
        pronunciation_feedback = "No pronunciation errors detected."
        
        self.turn_count += 1
        
        # Generate AI response
        coaching_action = CoachingAction(
            strategy=CoachingStrategy.FIND_EVIDENCE,
            target_skill="evidence_support",
            confidence_score=0.8
        )
        
        request = ResponseRequest(
            coaching_action=coaching_action,
            topic=topic or "renewable energy",
            user_position=user_position or transcript,
            prior_responses=self.prior_responses,
            turn_number=self.turn_count
        )
        
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.response_generator.generate_response(request)
            )
            loop.close()
            
            ai_response = response.text
            self.prior_responses.append(ai_response)
            if len(self.prior_responses) > 3:
                self.prior_responses.pop(0)
                
        except Exception as e:
            ai_response = f"Error generating response: {str(e)}"
        
        return (
            transcript,
            has_claim,
            has_reason, 
            has_evidence,
            argument_score,
            argument_summary,
            pronunciation_feedback,
            ai_response
        )

def create_interface():
    ui = SpeakFlowUI()
    
    with gr.Blocks(title="SpeakFlow AI - Debate Practice") as interface:
        gr.Markdown("# SpeakFlow AI - Debate Practice Prototype")
        gr.Markdown("Record your argument and get real-time feedback on structure and pronunciation.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Setup")
                topic_input = gr.Textbox(
                    label="Debate Topic",
                    placeholder="e.g., renewable energy policy",
                    value="renewable energy"
                )
                position_input = gr.Textbox(
                    label="Your Position", 
                    placeholder="e.g., I support renewable energy because...",
                    lines=2
                )
                
                gr.Markdown("### Record Your Argument")
                # Gradio 4.x: 'source' replaced by 'sources' (list)
                audio_input = gr.Audio(
                    label="Record Audio",
                    sources=["microphone"],
                    type="filepath"
                )
                
                analyze_btn = gr.Button("Analyze", variant="primary")
                
            with gr.Column(scale=2):
                gr.Markdown("### Transcript")
                transcript_output = gr.Textbox(
                    label="What you said",
                    lines=3,
                    interactive=False
                )
                
                gr.Markdown("### Argument Analysis")
                with gr.Row():
                    claim_check = gr.Checkbox(label="Has Clear Claim", interactive=False)
                    reason_check = gr.Checkbox(label="Has Reasoning", interactive=False)
                    evidence_check = gr.Checkbox(label="Has Evidence", interactive=False)
                
                argument_score = gr.Number(
                    label="Argument Score",
                    interactive=False
                )
                
                argument_summary = gr.Textbox(
                    label="Feedback Summary",
                    lines=2,
                    interactive=False
                )
                
                gr.Markdown("### Pronunciation Feedback")
                pronunciation_output = gr.Textbox(
                    label="Pronunciation Analysis",
                    lines=2,
                    interactive=False
                )
                
                gr.Markdown("### AI Response")
                ai_response_output = gr.Textbox(
                    label="Debate Partner Response",
                    lines=3,
                    interactive=False
                )
        
        analyze_btn.click(
            fn=ui.analyze_audio,
            inputs=[audio_input, topic_input, position_input],
            outputs=[
                transcript_output,
                claim_check,
                reason_check,
                evidence_check,
                argument_score,
                argument_summary,
                pronunciation_output,
                ai_response_output
            ]
        )
    
    return interface

if __name__ == "__main__":
    app = create_interface()
    app.launch(debug=True, share=False)