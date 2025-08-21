import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

# URL of the page
url = "https://www.vumc.org/trauma-and-scc/trauma-and-surgical-critical-care-practice-management-guidelines"
output_dir = "pdfs"
os.makedirs(output_dir, exist_ok=True)

# Get the page content
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# Find all PDF links
pdf_links = []
for a_tag in soup.find_all("a", href=True):
    href = a_tag['href']
    if href.lower().endswith(".pdf"):
        full_url = urljoin(url, href)
        pdf_links.append(full_url)

print(f"Found {len(pdf_links)} PDFs")

# Download PDFs
for link in pdf_links:
    filename = os.path.join(output_dir, link.split("/")[-1])
    print(f"Downloading {link} → {filename}")
    r = requests.get(link)
    with open(filename, "wb") as f:
        f.write(r.content)

print("All PDFs downloaded.")
