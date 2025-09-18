import gradio as gr
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
import re
import os


MODEL_NAME = "Salesforce/blip-image-captioning-base"

print(f"📦 Using model: {MODEL_NAME}")
print(f"💾 Models stored in: {os.path.expanduser('~/.cache/huggingface/transformers/')}")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize BLIP model
try:
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
    MODEL_LOADED = True
    print(f"✅ Loaded BLIP model on {device} (~400MB)")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    MODEL_LOADED = False
    model = None
    processor = None

def generate_caption_blip(image, style="detailed", max_length=50):
    """Generate caption using BLIP model"""
    try:
        inputs = processor(image, return_tensors="pt").to(device)
        
        # Different generation parameters for different styles
        if style == "simple":
            generated_ids = model.generate(**inputs, max_length=25, num_beams=3)
        elif style == "detailed":
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=4, do_sample=True, temperature=0.7)
        else:  # artistic
            generated_ids = model.generate(**inputs, max_length=max_length, num_beams=5, do_sample=True, temperature=0.9)
        
        caption = processor.decode(generated_ids[0], skip_special_tokens=True)
        return enhance_caption(caption, style)
        
    except Exception as e:
        return f"BLIP Error: {str(e)}"

def enhance_caption(caption, style):
    """Clean up and enhance captions"""
    # Remove common prefixes
    caption = re.sub(r'^(a|an|the|this is|there is)\s+', '', caption, flags=re.IGNORECASE)
    caption = re.sub(r'(image|picture|photo)\s+(of|showing)\s+', '', caption, flags=re.IGNORECASE)
    
    # Style enhancements
    if style == "simple":
        return caption.strip()
    elif style == "detailed":
        enhancers = ["detailed", "high-quality", "clear", "professional"]
        import random
        return f"{random.choice(enhancers)} {caption.strip()}"
    else:  # artistic
        enhancers = ["stunning", "beautiful", "artistic", "masterpiece", "breathtaking"]
        import random
        return f"{random.choice(enhancers)} {caption.strip()}"

def generate_lightweight_prompt(image, style="detailed", max_length=50):
    """Main function - always uses BLIP model if loaded, else fallback"""
    if not MODEL_LOADED:
        return generate_basic_fallback(image, style)
    try:
        return generate_caption_blip(image, style, max_length)
    except Exception as e:
        return generate_basic_fallback(image, style)

def generate_basic_fallback(image, style):
    """Ultra-basic fallback (no models needed)"""
    import numpy as np
    
    try:
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Basic analysis
        aspect_ratio = width / height
        if aspect_ratio > 1.5:
            comp = "wide landscape"
        elif aspect_ratio < 0.7:
            comp = "portrait orientation"
        else:
            comp = "balanced composition"
        
        # Color analysis
        if len(img_array.shape) == 3:
            avg_brightness = np.mean(img_array)
            if avg_brightness > 180:
                lighting = "bright, well-lit"
            elif avg_brightness > 100:
                lighting = "evenly lit"
            else:
                lighting = "moody, darker"
        else:
            lighting = "black and white"
        
        base_descriptions = {
            "simple": f"{comp} image",
            "detailed": f"A {lighting} {comp} photograph with good clarity",
            "artistic": f"Artistic {comp} with {lighting} atmospheric mood"
        }
        
        return base_descriptions.get(style, "processed image")
        
    except Exception as e:
        return f"Basic analysis: {style} style image"

def create_lightweight_interface():
    """Lightweight Gradio interface (no model info or delete instructions)"""
    
    with gr.Blocks(title="Lightweight Image-to-Prompt", theme=gr.themes.Monochrome()) as demo:
        gr.Markdown("# ⚡ Lightweight Image-to-Prompt")
        gr.Markdown("*Optimized for speed and storage efficiency*")
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    type="pil",
                    label="📷 Upload Image",
                    height=300
                )
                
                with gr.Row():
                    style_radio = gr.Radio(
                        choices=["simple", "detailed", "artistic"],
                        value="detailed",
                        label="Style",
                        scale=2
                    )
                    
                    length_slider = gr.Slider(
                        minimum=20,
                        maximum=80,
                        value=50,
                        step=10,
                        label="Length",
                        scale=1
                    )
                
                generate_btn = gr.Button("Generate", variant="primary")
                
            with gr.Column():
                prompt_output = gr.Textbox(
                    label="Generated Prompt",
                    lines=5,
                    placeholder="Upload an image..."
                )
                
                with gr.Row():
                    copy_btn = gr.Button("Copy", size="sm")
                    clear_btn = gr.Button("Clear", size="sm")
        
        # Event handlers
        def process_image(image, style, length):
            if image is None:
                return "Upload an image first!"
            return generate_lightweight_prompt(image, style, length)
        
        # Auto-generate on upload
        image_input.change(
            fn=process_image,
            inputs=[image_input, style_radio, length_slider],
            outputs=prompt_output
        )
        
        generate_btn.click(
            fn=process_image,
            inputs=[image_input, style_radio, length_slider],
            outputs=prompt_output
        )
        
        copy_btn.click(None, inputs=prompt_output, js="(text) => navigator.clipboard.writeText(text)")
        clear_btn.click(lambda: "", outputs=prompt_output)
    
    return demo

if __name__ == "__main__":
    print("🚀 Starting Lightweight Image-to-Prompt...")
    print(f"💾 Model cache: ~/.cache/huggingface/transformers/")
    
    demo = create_lightweight_interface()
    demo.launch()