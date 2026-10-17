#Code Grader# Code-Grader Setup and Execution

## 1. Activate the Environment

Activate the Conda environment named `code`:

```bash
conda activate code
```

## 2. Navigate to the Source Directory

Open the terminal in the project directory and navigate to the `src` folder:

```bash
cd src
```

> **Note:** Run this command from the `Code-Grader` project root directory.

## 3. Start the Application

Run the following command to start the Code-Grader application:

```bash
python grading_agent/main.py
```

This command starts the application and initiates the Llama server.

## 4. Stop the Server

To terminate the running server, press:

```text
Ctrl + C
```

## Troubleshooting

- **Conda environment not found:** Ensure that the Conda environment named `code` has been created.
- **Llama server fails to start:** Verify that the required Llama CLI is installed and available in your system's PATH.
- **Python module errors:** Ensure that the `code` environment is activated and the required project dependencies are installed.

