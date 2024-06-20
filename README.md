
# Multi-LLM Bot "Tobis"

## Overview

"Tobis" is an advanced chatbot designed to integrate multiple large language models (LLMs) to provide a versatile and comprehensive conversational experience. This project supports three LLMs—Llama3 by Meta, Gemini by Google, and Phi 3 (Mini)—and includes image generation capabilities via the Stable Diffusion model from Hugging Face.

## Features

- **Multi-Model Support**: Choose from Llama3, Gemini, or Phi 3 (Mini) for different conversational needs.
- **Image Generation**: Generate images based on prompts using the Stable Diffusion model.
- **Interactive Interface**: Built with Streamlit for a dynamic and user-friendly experience.
- **Session Management**: Maintains conversation context across interactions.

## Setup and Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/multi-llm-bot.git
   cd multi-llm-bot
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **API Keys**:
   - Add your Hugging Face and Google API keys to `secrets.toml`:
     ```toml
     hugging-api = "your-huggingface-api-key"
     gemini-api = "your-google-api-key"
     ```

4. **Run the application**:
   ```bash
   streamlit run chatbot.py
   ```

## Usage

- **Selecting Models**: Choose your preferred LLM from the sidebar.
- **Generating Text**: Input your prompts and receive responses from the selected model.
- **Generating Images**: Use the syntax `\img <description>` or `/img <description>` to generate images.

## Contributing

We welcome contributions to improve "Tobis". Please submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

By leveraging multiple LLMs and providing a rich set of features, "Tobis" aims to be a versatile and engaging conversational AI tool. We look forward to seeing how this project can be expanded and enhanced by the community.
