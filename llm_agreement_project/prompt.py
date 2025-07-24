
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
    
    return prompt_1

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
