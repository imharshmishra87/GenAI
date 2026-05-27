from langchain_text_splitters import RecursiveCharacterTextSplitter,Language


splitter=RecursiveCharacterTextSplitter.from_language(
    chunk_size=200,
    chunk_overlap=0,
    language=Language.PYTHON
)

text="""def greet_user(name: str) -> str:
    if not name:
        return "Hello, Mystery Guest!"
    return f"Hello, {name}! Welcome to Python."


# 1. Main execution block
if __name__ == "__main__":
    print("--- Starting Python Dummy Script ---")

    # 2. Variable assignments
    user_name = "Alice"
    data_list = [10, 25, 43, 66, 81, 100]
    even_numbers = []

    # 3. Call a function
    message = greet_user(user_name)
    print(message)

    # 4. Loop with conditional logic
    print("\nProcessing data list...")
    for number in data_list:
        if number % 2 == 0:
            even_numbers.append(number)
            print(f"-> Found an even number: {number}")
        else:
            print(f"-> Skipping odd number: {number}")

    # 5. Output results
    print("\n--- Summary ---")
    print(f"Original items processed: {len(data_list)}")
    print(f"Even numbers extracted: {even_numbers}")
    print("--- Script Finished Successfully ---")
"""

chunks=splitter.split_text(text)
print(chunks[0])