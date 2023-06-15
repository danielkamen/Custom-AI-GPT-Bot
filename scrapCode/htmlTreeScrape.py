from urllib.request import urlopen
from bs4 import BeautifulSoup

def extract_text(element, depth=0):
    if element.name is None:
        return str(element.string) if element.string else ''

    separator = '\n' + ' ' * depth if depth > 0 else ' '  # Add spaces for depth 0, tabs for others
    text = ''
    for child in element.children:
        if child.name == 'div' and 'thin' in child.get('class', []) and 'blue' in child.get('class', []) and 'lead' in child.get('class', []) and 'center' in child.get('class', []):
            sibling_tag = child.find_next_sibling('div', class_='thin blue lead center')
            sibling_text = sibling_tag.get_text(strip=True) if sibling_tag else ""
            if sibling_text == "Add to Cart":
                sibling_text = "This product is available"
            text += sibling_text + separator
        else:
            text += extract_text(child, depth + 1)

    return text.strip(separator)

html = urlopen('https://www.wachusett.com/tickets-passes/season-passes/season-passes/')
bs = BeautifulSoup(html.read(), 'html.parser')

main_content = bs.find('main')  # Find the <main> tag

if main_content:
    text = extract_text(main_content)

    # Save the output to a text file
    with open('output.txt', 'w', encoding='utf-8') as file:
        file.write(text)

    print("Output saved to 'output.txt'")
else:
    print("No <main> tag found on the webpage.")
