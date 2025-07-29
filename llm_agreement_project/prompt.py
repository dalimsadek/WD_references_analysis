
def author_type_prompt(ref_value, shots=None):
    """
    Generate a prompt for author type classification, optionally including few-shot examples.

    Parameters:
        ref_value (str): the reference to classify.
        shots (list of tuples): optional examples in the form [(ref, label), ...]
    """
    shot_str = ""
    if shots:
        for i, (shot_ref, shot_label) in enumerate(shots, 1):
            shot_str += f"""Example {i}:
Reference: {shot_ref}
Label: {shot_label}

"""
    prompt_1 = f"""
You are an expert assistant tasked with classifying the **type of author** for the following reference URL or citation:

Reference: {ref_value}

Please determine the author type based on the reference and choose one of the following categories:
- 'individual': if the content is authored by a named person.
- 'organisation': if the content is authored by a company, institution, or group.
- 'collective': if the article is signed by a collective group (e.g., editorial board).
- 'nw' (not well-defined): if the URL or metadata is broken, unclear, or not informative.
- 'ne' (not enough evidence): if there's not enough information to decide.
- 'dn' (does not apply): if the reference is irrelevant or malformed.


Your output should be **only one of these labels**: individual, organisation, collective, nw, ne, dn.
Return your answer as a JSON object using this format:
{{"label": "<one_of_the_labels_above>"}}

Only output the JSON. Do not include any explanations.
"""
    prompt_2_onesl = f"""
You are an expert assistant tasked with classifying the **type of author** for the following reference URL or citation.

{shot_str if shots else ""}Now classify this new reference:

Reference: {ref_value}

Please determine the author type based on the reference and choose one of the following categories:
- 'individual': if the content is authored by a named person.
- 'organisation': if the content is authored by a company, institution, or group.
- 'collective': if the article is signed by a collective group (e.g., editorial board).
+ 'collective': if the article is authored by a group of unnamed individuals acting together (e.g., editorial board, anonymous activist group), not tied to a formal institution.
- 'nw' (not well-defined): if the URL or metadata is broken, unclear, or not informative.
- 'ne' (not enough evidence): if there's not enough information to decide.
- 'dn' (does not apply): if the reference is irrelevant or malformed.

Your output should be **only one of these labels**: individual, organisation, collective, nw, ne, dn.
Return your answer as a JSON object using this format:
{{"label": "<one_of_the_labels_above>"}}

Only output the JSON. Do not include any explanations.
"""
    
    return prompt_2_onesl

def publisher_type_prompt(domain, examples=None):
    shots = ""
    if examples:
        shots = "\n\n".join([
            f"Domain: {ref}\nLabel: {lbl}"
            for ref, lbl in examples
        ]) + "\n\n"

    return f"""
You are evaluating the **type of publisher** associated with a web domain.

Your task is to classify a domain into one of the following categories:
- 'news': News outlet (e.g., BBC, CNN).
- 'company': Corporate website or brand.
- 'sp_source': Special interest group (e.g., activist organizations).
- 'academia': University, scholarly, or academic institution.
- 'govt': Government-related website.
- 'other': If it doesn't fit any of the above.
- 'nw' (not well-defined): if the domain is ambiguous or lacks proper information.

{shots}Now, classify this domain:
Domain: {domain}

Return your answer as a JSON object using this format:
{{"label": "<one_of_the_labels_above>"}}

Only output the JSON. Do not include any explanations.
"""


def publisher_verification_prompt(domain):
    return f"""
You are evaluating the **verification type** of the publisher associated with this domain:

Domain: {domain}

Choose the most appropriate label from the following options:
- 'yes': if the publisher is verified or reputable.
- 'no': if the publisher is known to be unreliable.
- 'vendor': if it's a commercial seller or online marketplace.
- 'no_profit': if the site belongs to a nonprofit organization.
- 'political': if the domain represents a political entity or campaign.
- 'cultural': if it's a cultural or artistic institution.
- 'trad_news': a traditional media outlet (e.g., established newspapers).
- 'non_trad_news': blogs, YouTube channels, or alt-media sites.
- 'academia_uni': university-level academic publisher.
- 'academia_pub': peer-reviewed academic journal or press.
- 'academia_other': other academic institution.
- 'nw': not well-defined.
- 'ne': not enough evidence to decide.
- 'dn': does not apply.

Please return one label from the list above.
Return your answer as a JSON object using this format:
{{"label": "<one_of_the_labels_above>"}}

Only output the JSON. 
STRICT INSTRUCTIONS :
Do not include any explanations.
"""


import requests
from bs4 import BeautifulSoup

def build_enriched_prompt(ref_url: str, task: str = "author_type") -> str:
    """
    Given a URL and task type, extracts page content + metadata and formats a task-specific prompt.
    Tasks supported: 'author_type', 'publisher_type', 'publisher_verification'.
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ChatGPT-PromptBuilder/1.0)"
        }
        resp = requests.get(ref_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"The reference is unreachable or malformed.\n\nReference: {ref_url}\n\nReturn this: {{\"label\": \"nw\"}}"

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Visible content: title + main paragraphs
    visible_text = soup.title.string.strip() if soup.title and soup.title.string else ""
    for p in soup.find_all("p"):
        txt = p.get_text(strip=True)
        if len(txt) > 100:
            visible_text += "\n" + txt
            break  # One good paragraph is usually enough

    # Metadata extraction
    metadata = {}
    for tag in soup.find_all("meta"):
        if tag.get("property", "").startswith("og:") or tag.get("name", "").startswith("twitter:"):
            key = tag.get("property") or tag.get("name")
            content = tag.get("content", "")
            metadata[key] = content

    # Add common metadata
    for name in ["description", "title", "author"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag:
            metadata[name] = tag.get("content", "")

    # Format metadata
    meta_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])

    # Prompt templates
    instruction = ""
    if task == "author_type":
        instruction = """
Your job is to determine the **type of author** for the given reference page.

Choose one of the following:
- 'individual': a named person.
- 'organisation': a company, institution, or group.
- 'collective': a group of unnamed individuals (e.g., editorial board).
- 'nw': not well-defined (e.g., broken or empty).
- 'ne': not enough evidence to decide.
- 'dn': does not apply.

Return a JSON with the format: {"label": "<your_label>"}
Only output the JSON. No explanation.
"""
    elif task == "publisher_type":
        instruction = """
Classify the **type of publisher** for the domain of the page.

Options:
- 'news': news outlet (e.g., BBC)
- 'company': corporate/brand website
- 'sp_source': activist or special interest group
- 'academia': university or scholarly
- 'govt': government site
- 'other': doesn't fit above
- 'nw': not well-defined

Return a JSON with the format: {"label": "<your_label>"}
Only output the JSON. No explanation.
"""
    elif task == "publisher_verification":
        instruction = """
Classify the **verification status** of the publisher.

Choose from:
- 'yes': if the publisher is verified or reputable.
- 'no': if the publisher is known to be unreliable.
- 'vendor': if it's a commercial seller or online marketplace.
- 'no_profit': if the site belongs to a nonprofit organization.
- 'political': if the domain represents a political entity or campaign.
- 'cultural': if it's a cultural or artistic institution.
- 'trad_news': a traditional media outlet (e.g., established newspapers).
- 'non_trad_news': blogs, YouTube channels, or alt-media sites.
- 'academia_uni': university-level academic publisher.
- 'academia_pub': peer-reviewed academic journal or press.
- 'academia_other': other academic institution.
- 'nw': not well-defined.
- 'ne': not enough evidence to decide.
- 'dn': does not apply.


Return a JSON with the format: {"label": "<your_label>"}
Only output the JSON. No explanation.
"""

    # Final prompt
    prompt = f"""You are given the following reference link:
{ref_url}

🔍 Page Title and Visible Content:
{visible_text.strip()}


🧾 Extracted Metadata:
{meta_str.strip()}


{instruction.strip()}
if you still not sure examine carefully this page : 
    {ref_url}"""

    return prompt

def build_relevance_prompt_from_ref(ref_url: str, item_label: str, property_label: str, value_label: str) -> str:
    """
    Build a relevance evaluation prompt for an LLM to decide if a reference supports a Wikidata statement.
    Returns a prompt formatted for JSON-only response: {"label": "0"} or {"label": "1"}
    """

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (compatible; ChatGPT-RelevanceEvaluator/1.0)"
        }
        resp = requests.get(ref_url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        return f"The reference is unreachable or malformed.\n\nReference: {ref_url}\n\nReturn this: {{\"label\": \"1\"}}"

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Visible content: title + one good paragraph
    visible_text = soup.title.string.strip() if soup.title and soup.title.string else ""
    for p in soup.find_all("p"):
        txt = p.get_text(strip=True)
        if len(txt) > 100:
            visible_text += "\n" + txt
            break

    # Metadata
    metadata = {}
    for tag in soup.find_all("meta"):
        if tag.get("property", "").startswith("og:") or tag.get("name", "").startswith("twitter:"):
            key = tag.get("property") or tag.get("name")
            content = tag.get("content", "")
            metadata[key] = content

    for name in ["description", "title", "author"]:
        tag = soup.find("meta", attrs={"name": name})
        if tag:
            metadata[name] = tag.get("content", "")

    meta_str = "\n".join([f"{k}: {v}" for k, v in metadata.items()])

    # Instruction
    instruction = f"""
You are given a Wikidata statement and a reference (web page). Your task is to evaluate whether the reference supports the statement.

🧾 Wikidata Statement:
- Item: {item_label}
- Property: {property_label}
- Value: {value_label}

🌐 Reference:
- Domain: {ref_url}

🎯 Task:
Determine if the reference **supports** the Wikidata statement.

Return `"label": "0"` if the reference **is relevant** (supports the statement).  
Return `"label": "1"` if the reference **is NOT relevant**.

Return a JSON with the format: 
```json
{{"label": "<your_label>"}}
Only output the JSON. No explanation. No other text.
"""
    prompt = f"""You are given the following reference link:
{ref_url}

🔍 Page Title and Visible Content:
{visible_text.strip()}

🧾 Extracted Metadata:
{meta_str.strip()}

{instruction.strip()}

If you are unsure, you may examine the page again manually:
{ref_url}
"""

    return prompt
