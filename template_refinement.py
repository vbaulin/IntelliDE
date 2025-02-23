import os
import re
from pathlib import Path

def extract_template_refinement(input_dir, output_file):
    """Extracts and combines Template Refinement sections from markdown files"""
    refinement_header = "## **M15: CT-GIN Template Self-Improvement Insights**"
    combined_content = [f"{refinement_header}\n---\n"]

    input_dir = Path(input_dir)
    output_file = Path(output_file)

    for md_file in input_dir.glob("**/*.md"):
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find the Template Refinement section
            section_match = re.search(
                rf'{re.escape(refinement_header)}(.*?)(?=\n## Summary|\n---\n|\Z)',
                content,
                re.DOTALL
            )

            if section_match:
                section_content = section_match.group(1).strip()
                # Add the section content to the combined output
                #combined_content.append(f"\n### From {md_file.name}\n")
                combined_content.append(f"{section_content}\n")

        except Exception as e:
            print(f"Error processing {md_file.name}: {str(e)}")

    # Write combined results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(combined_content)

if __name__ == "__main__":
    publications_dir = "../synthetic/publications6GIN8"
    output_md = "refinement.md"

    extract_template_refinement(publications_dir, output_md)
    print(f"Combined refinement sections saved to {output_md}")
