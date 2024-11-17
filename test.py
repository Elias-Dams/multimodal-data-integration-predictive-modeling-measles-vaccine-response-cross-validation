def remove_extra_spaces(input_file, output_file):
    try:
        # Open the input file and create the output file
        with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
            for line in infile:
                # Remove extra spaces and replace with a single space
                cleaned_line = ','.join(line.split())
                # Write the cleaned line to the output file
                outfile.write(cleaned_line + '\n')

        print(f"Extra spaces removed successfully. Output saved to '{output_file}'")

    except FileNotFoundError:
        print(f"Error: The file '{input_file}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    remove_extra_spaces('data/antibody_df.txt', 'data/antibody_df.csv')