import logging
from llm.server import initiate_server
from sandbox.docker_sandbox import DockerSandbox


logger = logging.getLogger(__name__)

def main():
    process = initiate_server()

    try:
        process.wait()

    except KeyboardInterrupt:
        logger.warning("\nStopping llama-server...")
        process.terminate()
        process.wait()

    finally:
      process.terminate()
    

    # sandbox = DockerSandbox()
    # sandbox.start()

    # try:
    #     sandbox.write_file(
    #         "hello.py",
    #         'print("Hello from the Docker sandbox!")\n',
    #     )

    #     result = sandbox.run(
    #         ["python", "hello.py"],
    #         timeout=5,
    #     )
    #     print("Exit code:", result.exit_code)
    #     print("Output:", result.output)

    #     # Give your agent access to sandbox.run,
    #     # sandbox.read_file, and sandbox.write_file.
    #     # Adapt this call to your actual LLM implementation.
    #     # initiate_server(sandbox=sandbox)

    # finally:
        # sandbox.close()
    
    process.close()



if __name__ == "__main__":
    main()