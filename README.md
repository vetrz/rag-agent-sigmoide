# Chapa Sigmoide RAG Agent

This repository hosts a Retrieval-Augmented Generation (RAG) agent designed to answer questions and provide information about the Legacy of "Chapa Sigmoide" electoral slate.

## How to Use

Follow these steps to set up and run the RAG agent on your local machine.

### Prerequisites

You need Git to clone the repository and Python 3.13 or superior installed on your system.

Dependency Manager: This project uses `uv`

**1. Clone the Repository**

First, clone the project from GitHub:

```
Bash

git clone https://github.com/vetrz/rag-agent-sigmoide.git
cd rag-agent-sigmoide
```

**2. Install Dependencies using `uv`**

Install uv and then use it to install the necessary Python packages:

Install `uv` (if you don't have it):

```
Bash

pip install uv
```

Install Project Dependencies:

```
Bash

uv install
```

## LLM Configuration Options

You have two options for using the Large Language Model (LLM) with this agent:

**Option A: Using the Gemini API (Recommended)**

To use the Google Gemini models, you need to provide your API key.

Get an API Key: Obtain your key from Google AI Studio.

Set the Key: You must set your Gemini API Key as an environment variable.

Method 1: `.env` file (Easiest for local development) Create a file named `.env` in the root directory of the cloned project and add your key:

```
GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

Method 2: System Environment Variable Set the environment variable directly in your terminal before running the script:

```
Bash

# For Linux/macOS
export GEMINI_API_KEY="YOUR_API_KEY_HERE"

# For Windows (Command Prompt)
set GEMINI_API_KEY="YOUR_API_KEY_HERE"
```

**Option B: Using Ollama for Local Models**

You can install Ollama using a single command in your terminal. This is the recommended method for Linux and macOS.

**1. Execute the Installation Command**

Open your terminal and run the following command:

```
Bash

curl -fsSL https://ollama.com/install.sh | sh
```

**2. Verify Installation**

After the script finishes, Ollama will automatically start running in the background. Check the version to confirm the installation:

```
Bash

ollama --version
```

**3. Run a Local Model**

Use the ollama run command to download and start interacting with a model immediately (e.g., Llama 3):

```
Bash

ollama run llama3
```

Ensure Ollama is Running: Make sure the Ollama server is running in the background before execution.

## Running the Agent

Once you've configured your chosen LLM option and installed dependencies, run the main script.

Execute the Code

Run the agent script using uv Python:

```
Bash

uv run -m rag_agent.main

```

The script will initialize the RAG process using the documents specific to the "Chapa Sigmoide" and allow you to interact with the LLM.
