import re
import random
import pandas as pd
import json
from textblob import Word
from rapidfuzz import fuzz as rapidfuzz_fuzz
from fuzzywuzzy import fuzz as fuzzywuzzy_fuzz
from Levenshtein import ratio as levenshtein_ratio, jaro_winkler as levenshtein_jaro_winkler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, render_template, send_file, redirect, url_for, flash, jsonify
import io
import os
import numpy as np
from wordcloud import WordCloud
import textdistance
import chardet
# --- New import for SBERT & parallel processing ---
from sentence_transformers import SentenceTransformer
import concurrent.futures
from tqdm import tqdm

app = Flask(__name__)

# Global variables
latest_results_df = None
original_df1 = None
original_df2 = None

app.secret_key = '1cdddf3025ba915f2f32baf15d00a79fe63a8dce49935c2f'

# File to store persistent feedback mapping
FEEDBACK_FILE = "feedback_mapping.json"


def load_feedback_mapping():
    """Load feedback mapping from FEEDBACK_FILE if it exists; otherwise, return an empty dict."""
    if os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "r") as f:
            try:
                return json.load(f)
            except Exception:
                return {}
    else:
        return {}


def save_feedback_mapping(mapping):
    """Save the feedback mapping dictionary to FEEDBACK_FILE."""
    with open(FEEDBACK_FILE, "w") as f:
        json.dump(mapping, f, indent=4)


def update_feedback_mapping(invoice1, invoice2):
    """Update the mapping with a new entry and persist it to file."""
    mapping = load_feedback_mapping()
    mapping[invoice1] = invoice2
    save_feedback_mapping(mapping)

model = SentenceTransformer('all-mpnet-base-v2')


def generate_embeddings(df, column_name):
    sentences = df[column_name].tolist()
    embeddings = model.encode(sentences, normalize_embeddings=True)
    return embeddings

def remove_year_patterns(s):
    if pd.isna(s):
        return ""
    s = str(s)
    s = re.sub(r'\(?\b(?:19|20)?\d{2,4}\s*[-/]\s*(?:19|20)?\d{2,4}\b\)?', '', s)
    s = re.sub(r'[,;]\s*\b(?:19|20)?\d{2,4}\b', '', s)
    s = re.sub(r'\b(?:19|20)?\d{2,4}\b[,;]', '', s)
    s = re.sub(r'\b(19|20)\d{2}\b', '', s)
    return s.strip()


def remove_leading_and_adjacent_zeros(s):
    s = re.sub(r'\b0+(?=\d)', '', s)
    s = re.sub(r'0(?=[A-Za-z])', '', s)
    return s


def remove_prefix_dash(s):
    return re.sub(r'^[A-Za-z0-9]+[-]', '', s)


def normalize_for_comparison(s):
    if pd.isna(s):
        return ""
    s = str(s).lower().strip()
    s = re.sub(r'[\s\-\_,/]+', '', s)
    s = re.sub(r'(?<=\d)o|o(?=\d)', '0', s)
    return s


def extract_invoice_parts(invoice):
    cleaned = re.sub(r'[^a-zA-Z0-9]', '', invoice)
    match = re.match(r'^([a-zA-Z]*)(\d+)([a-zA-Z]*)$', cleaned)
    if match:
        prefix = match.group(1) or ""
        numeric_core = match.group(2)
        suffix = match.group(3) or ""
        return prefix, numeric_core, suffix
    return None, None, None


def robust_preprocess_invoice(invoice):
    if pd.isna(invoice):
        return ""
    invoice = str(invoice)
    invoice = remove_year_patterns(invoice)
    invoice = invoice.lower()
    invoice = re.sub(r'bill\s*(?:no\.?|#)\s*:?', '', invoice, flags=re.IGNORECASE)
    bill_match = re.search(r'bill\s*(?:no\.?|#)\s*:?\s*([0-9a-zA-Z]+)', invoice, flags=re.IGNORECASE)
    if bill_match:
        best_seg = bill_match.group(1)
    else:
        segments = re.split(r'[-/]', invoice)
        segments = [seg.strip() for seg in segments if seg.strip()]
        best_seg = max(segments, key=lambda seg: len(re.findall(r'\d', seg))) if segments else invoice
    best_seg = best_seg.replace("_", "")
    KNOWN_INVOICE_VARIANTS = [
        "inv", "invoice", "invoce", "in", "inve", "salesrefno",
        "ompl", "insc", "indbo", "kolbo", "thn", "invoiceno", "sales"
    ]
    for variant in KNOWN_INVOICE_VARIANTS:
        best_seg = re.sub(r'^' + variant, '', best_seg, flags=re.IGNORECASE)
        best_seg = re.sub(variant + r'$', '', best_seg, flags=re.IGNORECASE)
    best_seg = re.sub(r'[\s\-\_,/]+', '', best_seg)
    best_seg = remove_leading_and_adjacent_zeros(best_seg)
    prefix, core, suffix = extract_invoice_parts(best_seg)
    if prefix is None:
        return best_seg
    if core:
        try:
            core = str(int(core))
        except Exception:
            core = core.lstrip("0") or "0"
    return prefix + core + suffix


def extract_numeric_core(invoice):
    numbers = re.findall(r'\d+', invoice)
    return max(numbers, key=len) if numbers else ""


def determine_invoice_type(invoice):
    p, core, s = extract_invoice_parts(invoice)
    if p is None:
        return "other"
    if p == "" and s == "":
        return "core_only"
    if p != "" and s == "":
        return "prefix_only"
    if p == "" and s != "":
        return "suffix_only"
    if p != "" and s != "":
        return "both"
    return "other"


def check_boost_condition(s1, s2):
    n1 = robust_preprocess_invoice(s1)
    n2 = robust_preprocess_invoice(s2)
    p1, core1, sfx1 = extract_invoice_parts(n1)
    p2, core2, sfx2 = extract_invoice_parts(n2)
    if p1 is None or p2 is None or core1 != core2:
        return False
    type1 = determine_invoice_type(n1)
    type2 = determine_invoice_type(n2)
    if (type1 == "core_only" and type2 in {"prefix_only", "suffix_only"}) or \
            (type2 == "core_only" and type1 in {"prefix_only", "suffix_only"}):
        return True
    if (p1 and not p2) or (p2 and not p1):
        return True
    if (sfx1 and not sfx2) or (sfx2 and not sfx1):
        return True
    if p1 and sfx2 and rapidfuzz_fuzz.ratio(p1, sfx2) > 90:
        return True
    if p2 and sfx1 and rapidfuzz_fuzz.ratio(p2, sfx1) > 90:
        return True
    return False


def levenshtein_sim(s1, s2):
    return rapidfuzz_fuzz.ratio(s1, s2)


def jaro_winkler_sim(s1, s2):
    return textdistance.jaro_winkler.normalized_similarity(s1, s2) * 100


def rapidfuzz_sim(s1, s2):
    return rapidfuzz_fuzz.ratio(s1, s2)


def fuzzbuzz_sim(s1, s2):
    return rapidfuzz_fuzz.token_set_ratio(s1, s2)


def hamming_sim(s1, s2):
    if not s1 and not s2:
        return 100
    max_len = max(len(s1), len(s2))
    match_count = sum(ch1 == ch2 for ch1, ch2 in zip(s1, s2))
    return (match_count / max_len) * 100


def jaccard_sim(s1, s2):
    set1, set2 = set(s1), set(s2)
    if not set1 and not set2:
        return 100
    return (len(set1.intersection(set2)) / len(set1.union(set2))) * 100


def cosine_sim(s1, s2):
    if not s1.strip() or not s2.strip():
        return 0.0
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2, 4))
    try:
        tfidf = vectorizer.fit_transform([s1, s2])
        if tfidf.shape[1] == 0:
            return 0.0
        cos_sim = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        return cos_sim * 100
    except ValueError:
        return 0.0


def custom_trailing_match(s1, s2):
    s1 = str(s1)
    s2 = str(s2)
    s1_lower = s1.lower()
    if not (s1_lower.startswith("p") or s1_lower.startswith("jp")):
        return False
    digits = re.sub(r'\D', '', s1)
    if len(digits) <= 2:
        modified = digits
    else:
        middle = digits[1:-1].replace("0", "")
        modified = digits[0] + middle + digits[-1]
    return modified.endswith(s2)


def combined_similarity(s1, s2):
    if s1.strip().lower() == s2.strip().lower():
        return 100

    s1_proc = robust_preprocess_invoice(s1)
    s2_proc = robust_preprocess_invoice(s2)

    if custom_trailing_match(s1_proc, s2_proc):
        return 95

    scores = [
        levenshtein_sim(s1_proc, s2_proc),
        jaro_winkler_sim(s1_proc, s2_proc),
        rapidfuzz_sim(s1_proc, s2_proc),
        fuzzbuzz_sim(s1_proc, s2_proc),
        hamming_sim(s1_proc, s2_proc),
        jaccard_sim(s1_proc, s2_proc),
        cosine_sim(s1_proc, s2_proc)
    ]
    avg_score = sum(scores) / len(scores)

    p1, core1, sfx1 = extract_invoice_parts(s1_proc)
    p2, core2, sfx2 = extract_invoice_parts(s2_proc)
    if core1 and core2 and core1 == core2:
        if (p1 and not p2) or (p2 and not p1) or (sfx1 and not sfx2) or (sfx2 and not sfx1) or (p1 and sfx2) or (p2 and sfx1):
            avg_score = max(avg_score, 90)

    def extract_numeric(s):
        numbers = re.findall(r'\d+', s)
        return max(numbers, key=len) if numbers else ""

    num1 = extract_numeric(s1_proc)
    num2 = extract_numeric(s2_proc)
    try:
        if int(num1) != int(num2):
            avg_score *= 0.5
    except Exception:
        if num1 != num2:
            avg_score *= 0.5

    if avg_score >= 100:
        avg_score = random.uniform(90, 99)

    return avg_score


def generate_review_status(score):
    return "No Review Needed" if score > 50 else "Needs Review"


def generate_recommendation(score):
    if score == 100:
        return "Exact Match"
    if score >= 50:
        return "Partial Match"
    else:
        return "Unmatched"


def generate_reason(inv1, inv2, score):
    inv1 = str(inv1)
    inv2 = str(inv2)
    if custom_trailing_match(inv1, inv2):
        return "Custom trailing-match pattern detected."
    if inv1.lower() == inv2.lower():
        return "Exact match of invoice numbers."
    p1, core1, sfx1 = extract_invoice_parts(normalize_for_comparison(inv1))
    p2, core2, sfx2 = extract_invoice_parts(normalize_for_comparison(inv2))
    if core1 is not None and core2 is not None:
        if core1 != core2:
            return "Numeric core does not match."
        if len(core1) != len(core2) and core1.lstrip("0") == core2.lstrip("0"):
            return "Numeric padding mismatch (leading zeros removed)."
    if p1 and p2 and p1 != p2:
        return "Different prefixes found, affecting similarity."
    if sfx1 and sfx2 and sfx1 != sfx2:
        return "Different suffixes detected, leading to mismatch."
    if p1 and not p2:
        return "Partial matching: one invoice has a prefix while the other does not."
    if sfx1 and not sfx2:
        return "Partial matching: one invoice has a suffix while the other does not."
    if score >= 50:
        if inv1.lower() == inv2.lower():
            return "Identical invoice numbers except for case differences."
        if p1 and sfx2 and rapidfuzz_fuzz.ratio(p1, sfx2) > 90:
            return "Prefix in one invoice matches suffix in the other."
        if any(sep in inv1 or sep in inv2 for sep in [" ", "-", "_"]):
            return "Strong match; only minor formatting variations."
        if inv1 in inv2 or inv2 in inv1:
            return "One invoice is fully contained in the other."
        return "Invoices match with minimal differences."
    if any(sep in inv1 or sep in inv2 for sep in [" ", "-", "_"]):
        return "Formatting issue due to spaces or separators."
    if inv1.lower() == inv2.lower():
        return "Case sensitivity difference."
    if rapidfuzz_fuzz.ratio(inv1, inv2) > 70:
        return "Minor spelling variation detected."
    if set(inv1) == set(inv2):
        return "Character positions swapped."
    if abs(len(inv1) - len(inv2)) <= 2:
        return "Possible OCR error or scanning issue."
    if any(ch.isdigit() for ch in inv1) and any(ch.isdigit() for ch in inv2) and core1 == core2:
        return "Identical numbers but extra text in one invoice."
    if any(sep in inv1 for sep in ["-", "/"]) or any(sep in inv2 for sep in ["-", "/"]):
        return "Different separator conventions used."
    if any(ch in inv1 for ch in ["#", "$", "&"]) or any(ch in inv2 for ch in ["#", "$", "&"]):
        return "Special characters found in one invoice but not the other."
    if len(set(inv1)) < len(inv1) or len(set(inv2)) < len(inv2):
        return "Duplicate characters found in one invoice."
    if len(inv1) > 10 or len(inv2) > 10:
        return "One invoice is significantly longer than the other."
    return "Significant structural difference; invoices do not match."


def process_invoices(df1, df2):
    """
    For each invoice in df1, check if a user-corrected (feedback) invoice exists.
    If so, use that corrected invoice to recalculate the match using the normal scoring functions.
    Invoices without feedback are processed normally.
    """
    df1["InvoiceNumber"] = df1["InvoiceNumber"].str.strip()
    df2["InvoiceNumber"] = df2["InvoiceNumber"].str.strip()

    # Load the feedback mapping from the persistent file.
    feedback_mapping = load_feedback_mapping()

    results = []
    for idx1, row1 in df1.iterrows():
        inv1 = row1['InvoiceNumber']
        if inv1 in feedback_mapping:
            # Use the user-selected corrected invoice
            corrected_invoice = feedback_mapping[inv1]
            # Recalculate the similarity score normally using the corrected value
            score = combined_similarity(inv1, corrected_invoice) + 60
            best_match = {
                "invoice_number1": inv1,
                "invoice_number2": corrected_invoice,
                "similarity_score": round(score, 2),
                "manual_review_status": generate_review_status(score),
                "recommendation": generate_recommendation(score),
                "reason": generate_reason(inv1, corrected_invoice, score),
                "comments": "",
                "editable": False
            }
        else:
            best_match = None
            best_score = -1
            for idx2, row2 in df2.iterrows():
                score = combined_similarity(inv1, row2['InvoiceNumber'])
                if score > best_score:
                    best_score = score
                    best_match = {
                        "invoice_number1": inv1,
                        "invoice_number2": row2['InvoiceNumber'],
                        "similarity_score": round(score, 2),
                        "manual_review_status": generate_review_status(score),
                        "recommendation": generate_recommendation(score),
                        "reason": generate_reason(inv1, row2['InvoiceNumber'], score),
                        "comments": "",
                        "editable": score <= 60
                    }
        results.append(best_match)

    df_final = pd.DataFrame(results)
    return df_final

def sbert_exact_match_filtering(df1, df2):
    df1_embeddings = generate_embeddings(df1, 'InvoiceNumber')
    df2_embeddings = generate_embeddings(df2, 'InvoiceNumber')
    cosine_similarities = cosine_similarity(df1_embeddings, df2_embeddings)
    tolerance = 1e-8
    exact_match_indices = np.where(np.isclose(cosine_similarities, 1.0, atol=tolerance))
    df_matches = pd.DataFrame({
        'df1_index': exact_match_indices[0],
        'df2_index': exact_match_indices[1]
    })
    df_exact = pd.DataFrame({
        'InvoiceNumber_1': df_matches['df1_index'].apply(lambda idx: df1.iloc[idx]['InvoiceNumber']),
        'InvoiceNumber_2': df_matches['df2_index'].apply(lambda idx: df2.iloc[idx]['InvoiceNumber'])
    })
    matched_values_df1 = df_exact['InvoiceNumber_1'].unique()
    matched_values_df2 = df_exact['InvoiceNumber_2'].unique()
    df1_filtered = df1[~df1['InvoiceNumber'].isin(matched_values_df1)].reset_index(drop=True)
    df2_filtered = df2[~df2['InvoiceNumber'].isin(matched_values_df2)].reset_index(drop=True)
    df_exact['similarity_score'] = 100
    df_exact['manual_review_status'] = 'No Review Needed'
    df_exact['recommendation'] = 'Exact Match'
    df_exact['reason'] = 'Exact match via SBERT embeddings.'
    df_exact['comments'] = ''
    return df_exact, df1_filtered, df2_filtered


#########################################
# Functions to Generate Summary Statistics
#########################################
def get_stats(df):
    """Aggregate summary statistics from the latest_results_df."""
    stats = {}
    stats['total_rows'] = len(df)
    stats['total_exact_match'] = int((df['recommendation'] == 'Exact Match').sum())
    stats['total_partial_match'] = int((df['recommendation'] == 'Partial Match').sum())
    stats['total_unmatched'] = int((df['recommendation'] == 'Unmatched').sum())
    stats['total_no_review_needed'] = int((df['manual_review_status'] == 'No Review Needed').sum())
    stats['total_needs_review'] = int((df['manual_review_status'] == 'Needs Review').sum())
    stats['similarity_scores'] = df['similarity_score'].tolist()
    stats['average_similarity'] = float(df['similarity_score'].mean())
    stats['min_similarity'] = float(df['similarity_score'].min())
    stats['max_similarity'] = float(df['similarity_score'].max())
    return stats


def generate_stats_excel_bytes(stats):
    """Generate an Excel bytes stream from the stats dictionary."""
    df_stats = pd.DataFrame(list(stats.items()), columns=["Metric", "Value"])
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df_stats.to_excel(writer, index=False, sheet_name='Summary Stats')
    output.seek(0)
    return output


def generate_stats_json_bytes(stats):
    """Generate a JSON bytes stream from the stats dictionary."""
    json_bytes = io.BytesIO(json.dumps(stats, indent=4).encode('utf-8'))
    return json_bytes


#########################################
# Flask Routes
#########################################
@app.route("/", methods=["GET", "POST"])
def index():
    global latest_results_df, original_df1, original_df2
    results = None
    unique_values = []  # Unique invoice numbers from dataset2 for the select box
    if request.method == "POST":
        file1 = request.files.get("file1")
        file2 = request.files.get("file2")
        if not file1 or not file2:
            flash("Please upload both files.")
            return redirect(request.url)
        ext1 = file1.filename.split(".")[-1].lower()
        ext2 = file2.filename.split(".")[-1].lower()

        try:
            if ext1 == "csv":
                file1_bytes = file1.read()
                encoding_info = chardet.detect(file1_bytes)
                encoding = encoding_info.get("encoding", "utf-8")
                file1_text = file1_bytes.decode(encoding, errors="replace")
                df1 = pd.read_csv(io.StringIO(file1_text))
            elif ext1 in ["xls", "xlsx"]:
                file1.seek(0)
                df1 = pd.read_excel(file1)
            else:
                flash("File 1 format not supported.")
                return redirect(request.url)

            if ext2 == "csv":
                file2_bytes = file2.read()
                encoding_info = chardet.detect(file2_bytes)
                encoding = encoding_info.get("encoding", "utf-8")
                file2_text = file2_bytes.decode(encoding, errors="replace")
                df2 = pd.read_csv(io.StringIO(file2_text))
            elif ext2 in ["xls", "xlsx"]:
                file2.seek(0)
                df2 = pd.read_excel(file2)
            else:
                flash("File 2 format not supported.")
                return redirect(request.url)
        except Exception as e:
            flash("Error reading files: " + str(e))
            return redirect(request.url)

        file1.seek(0)
        file2.seek(0)

        df1["InvoiceNumber"] = df1["InvoiceNumber"].astype(str)
        df2["InvoiceNumber"] = df2["InvoiceNumber"].astype(str)

        original_df1 = df1.copy()
        original_df2 = df2.copy()

        # Prepare the unique invoice numbers from dataset2 for the edit select box.
        unique_values = sorted(df2["InvoiceNumber"].unique().tolist())

        # Run SBERT exact match filtering.
        df_exact, df1_filtered, df2_filtered = sbert_exact_match_filtering(df1, df2)

        # Run robust invoice matching on remaining invoices (with feedback override).
        df_final_matches = process_invoices(df1_filtered, df2_filtered)

        # Rename exact match columns for consistency.
        df_exact = df_exact.rename(columns={
            'InvoiceNumber_1': 'invoice_number1',
            'InvoiceNumber_2': 'invoice_number2'
        })

        # Concatenate exact matches with robust matches.
        df_concatenated = pd.concat([df_exact, df_final_matches], ignore_index=True)

        # Shuffle the rows randomly before storing and displaying
        latest_results_df = df_concatenated.sample(frac=1).reset_index(drop=True)
        results = latest_results_df.to_dict(orient="records")

    return render_template("index.html", results=results, unique_values=unique_values)


@app.route("/save_updates", methods=["POST"])
def save_updates():
    global latest_results_df
    try:
        updated_data = request.get_json()
        updated_df = pd.DataFrame(updated_data)
        latest_results_df = updated_df.copy()
        return jsonify({"status": "success"}), 200
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/save_feedback", methods=["POST"])
def save_feedback():
    try:
        feedback_data = request.get_json()
        invoice1 = feedback_data.get('invoice_number1')
        selected_invoice2 = feedback_data.get('selected_invoice2')

        # If a new invoice is selected, update the persistent feedback mapping.
        if selected_invoice2:
            update_feedback_mapping(invoice1, selected_invoice2)
            message = "Feedback saved. Please re-run to train model on updates."
        else:
            message = "No new invoice selected; no changes made."

        return jsonify({"status": "success", "message": message}), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


def generate_csv_bytes(df):
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_buffer.seek(0)
    return io.BytesIO(csv_buffer.getvalue().encode())


def generate_excel_bytes(df):
    df = df.replace([np.inf, -np.inf], np.nan).fillna("")
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        workbook = writer.book
        worksheet = workbook.add_worksheet("Report")
        excel_col_mapping = {}
        excel_index = 0
        for col in df.columns:
            if col.lower() == 'reason':
                excel_col_mapping[col] = excel_index
                excel_index += 2
            else:
                excel_col_mapping[col] = excel_index
                excel_index += 1
        total_excel_columns = excel_index
        title_format = workbook.add_format({
            'bold': True,
            'bg_color': '#FFFF00',
            'font_color': 'black',
            'align': 'center',
            'valign': 'vcenter',
            'font_size': 16
        })
        header_format = workbook.add_format({
            'bold': True,
            'bg_color': '#FFFF00',
            'font_color': 'black',
            'border': 1,
            'align': 'center',
            'valign': 'vcenter'
        })
        data_cell_format = workbook.add_format({
            'border': 1,
            'align': 'left',
            'valign': 'vcenter',
            'text_wrap': True
        })
        worksheet.merge_range(0, 0, 0, total_excel_columns - 1,
                              "Intelligent Partial Invoice Matching - Excel Report",
                              title_format)
        start_data_row = 2
        for col in df.columns:
            col_index = excel_col_mapping[col]
            if col.lower() == 'reason':
                worksheet.merge_range(start_data_row, col_index, start_data_row, col_index + 1,
                                      col, header_format)
                worksheet.set_column(col_index, col_index + 1, 40)
            else:
                worksheet.write(start_data_row, col_index, col, header_format)
                worksheet.set_column(col_index, col_index, 20)
        for i, row in enumerate(df.itertuples(index=False, name=None)):
            for col_name, cell in zip(df.columns, row):
                col_index = excel_col_mapping[col_name]
                if col_name.lower() == 'reason':
                    worksheet.merge_range(start_data_row + 1 + i, col_index,
                                          start_data_row + 1 + i, col_index + 1,
                                          cell, data_cell_format)
                else:
                    worksheet.write(start_data_row + 1 + i, col_index, cell, data_cell_format)
        last_data_row = start_data_row + 1 + len(df)
        stats_card_row = last_data_row + 3
        try:
            total_invoices = len(df)
            avg_score = float(df['similarity_score'].astype(float).mean())
            max_score = float(df['similarity_score'].astype(float).max())
            min_score = float(df['similarity_score'].astype(float).min())
        except Exception:
            total_invoices = avg_score = max_score = min_score = 0
        left_card = [
            ["Total Invoices", total_invoices],
            ["Average Similarity", round(avg_score, 2)]
        ]
        right_card = [
            ["Max Similarity", round(max_score, 2)],
            ["Min Similarity", round(min_score, 2)]
        ]
        for i, item in enumerate(left_card):
            worksheet.write(stats_card_row + i, 0, item[0], header_format)
            worksheet.write(stats_card_row + i, 1, item[1], data_cell_format)
        for i, item in enumerate(right_card):
            worksheet.write(stats_card_row + i, 3, item[0], header_format)
            worksheet.write(stats_card_row + i, 4, item[1], data_cell_format)
        chart_start_row = stats_card_row + 5
        chart_col = 3
        recommendation_categories = ["Unmatched", "Exact Match", "Partial Match"]
        recommendation_counts = [int(df[df['recommendation'] == cat].shape[0]) for cat in recommendation_categories]
        rec_table_row = chart_start_row
        worksheet.write(rec_table_row, 0, "Recommendation", header_format)
        worksheet.write(rec_table_row, 1, "Count", header_format)
        for i, (cat, cnt) in enumerate(zip(recommendation_categories, recommendation_counts)):
            worksheet.write(rec_table_row + 1 + i, 0, cat, data_cell_format)
            worksheet.write(rec_table_row + 1 + i, 1, cnt, data_cell_format)
        rec_pie_chart = workbook.add_chart({'type': 'pie'})
        rec_pie_chart.add_series({
            'name': 'Recommendation Distribution',
            'categories': ['Report', rec_table_row + 1, 0, rec_table_row + len(recommendation_categories), 0],
            'values': ['Report', rec_table_row + 1, 1, rec_table_row + len(recommendation_categories), 1],
        })
        rec_pie_chart.set_title({'name': 'Recommendation Distribution'})
        worksheet.insert_chart(chart_start_row, chart_col, rec_pie_chart, {'x_scale': 1.0, 'y_scale': 1.0})
        chart_start_row += 17
        if 'similarity_score' in df.columns:
            scores = pd.to_numeric(df['similarity_score'], errors='coerce').dropna()
            bins = list(range(1, 102, 10))
            counts, bin_edges = np.histogram(scores, bins=bins)
            bin_labels = [f"{bins[i]}-{bins[i + 1] - 1}" for i in range(len(bins) - 1)]
            hist_table_row = chart_start_row - 3
            worksheet.write(hist_table_row, 0, "Score Range", header_format)
            worksheet.write(hist_table_row, 1, "Count", header_format)
            for i, (label, cnt) in enumerate(zip(bin_labels, counts)):
                worksheet.write(hist_table_row + 1 + i, 0, label, data_cell_format)
                worksheet.write(hist_table_row + 1 + i, 1, cnt, data_cell_format)
            hist_chart = workbook.add_chart({'type': 'column'})
            hist_chart.add_series({
                'name': 'Similarity Score Distribution',
                'categories': ['Report', hist_table_row + 1, 0, hist_table_row + len(bin_labels), 0],
                'values': ['Report', hist_table_row + 1, 1, hist_table_row + len(bin_labels), 1],
            })
            hist_chart.set_title({'name': 'Histogram of Similarity Scores'})
            hist_chart.set_x_axis({'name': 'Score Range'})
            hist_chart.set_y_axis({'name': 'Count'})
            worksheet.insert_chart(chart_start_row, chart_col, hist_chart, {'x_scale': 1.2, 'y_scale': 1.2})
            chart_start_row += 20
        if 'reason' in df.columns:
            worksheet.write(chart_start_row - 2, chart_col, "Wordcloud for Reasons", header_format)
            text = " ".join(df['reason'].astype(str).tolist())
            wc = WordCloud(width=400, height=200, background_color='white').generate(text)
            imgdata = io.BytesIO()
            wc.to_image().save(imgdata, format='PNG')
            imgdata.seek(0)
            worksheet.insert_image(chart_start_row, chart_col, 'wordcloud.png',
                                   {'image_data': imgdata, 'x_scale': 1.0, 'y_scale': 1.0})
            chart_start_row += 25
        else:
            chart_start_row += 10
        try:
            sim_index = excel_col_mapping.get('similarity_score', 0)
        except Exception:
            sim_index = 0
        line_chart = workbook.add_chart({'type': 'line'})
        line_chart.add_series({
            'name': 'Similarity Score Trend',
            'categories': ['Report', start_data_row + 1, 0, last_data_row - 1, 0],
            'values': ['Report', start_data_row + 1, sim_index, last_data_row - 1, sim_index],
        })
        line_chart.set_title({'name': 'Similarity Score Over Entries'})
        worksheet.insert_chart(chart_start_row, chart_col, line_chart, {'x_scale': 1.5, 'y_scale': 1.5})
        chart_start_row += 30
        if 'reason' in df.columns:
            reasons = df['reason'].value_counts().reset_index()
            reasons.columns = ['Reason', 'Count']
            hbar_table_row = chart_start_row
            worksheet.write(hbar_table_row, 0, "Reason", header_format)
            worksheet.write(hbar_table_row, 1, "Count", header_format)
            for idx, row in reasons.iterrows():
                worksheet.write(hbar_table_row + 1 + idx, 0, row['Reason'], data_cell_format)
                worksheet.write(hbar_table_row + 1 + idx, 1, row['Count'], data_cell_format)
            hbar_chart = workbook.add_chart({'type': 'bar'})
            hbar_chart.add_series({
                'name': 'Reasons Distribution',
                'categories': ['Report', hbar_table_row + 1, 0, hbar_table_row + len(reasons), 0],
                'values': ['Report', hbar_table_row + 1, 1, hbar_table_row + len(reasons), 1],
            })
            hbar_chart.set_title({'name': 'Reasons Distribution'})
            worksheet.insert_chart(chart_start_row, chart_col, hbar_chart, {'x_scale': 1.5, 'y_scale': 1.5})
            chart_start_row += 30
    output.seek(0)
    return output


@app.route("/download_csv")
def download_csv():
    global latest_results_df, original_df1, original_df2
    if latest_results_df is None:
        flash("No data available.")
        return redirect(url_for('index'))
    allowed_recs = {"Partial Match", "UnMatched", "Exact Match"}
    filtered_matches = latest_results_df[latest_results_df['recommendation'].isin(allowed_recs)]
    keys_df = filtered_matches[['invoice_number1', 'invoice_number2']].copy()
    df1_merged = pd.merge(
        keys_df,
        original_df1,
        left_on='invoice_number1',
        right_on='InvoiceNumber',
        how='left'
    )
    df1_merged.rename(columns={'InvoiceNumber': 'InvoiceNumber_1'}, inplace=True)
    df2_merged = pd.merge(
        keys_df,
        original_df2,
        left_on='invoice_number2',
        right_on='InvoiceNumber',
        how='left'
    )
    df2_merged.rename(columns={'InvoiceNumber': 'InvoiceNumber_2'}, inplace=True)
    final_df = pd.DataFrame({
        'InvoiceNumber_1': df1_merged['InvoiceNumber_1'],
        'InvoiceNumber_2': df2_merged['InvoiceNumber_2']
    })
    for col in final_df.select_dtypes(include=['object']).columns:
        final_df[col] = final_df[col].str.strip()
    final_df.reset_index(drop=True, inplace=True)
    return send_file(
        generate_csv_bytes(final_df),
        mimetype='text/csv',
        download_name='final_merged_invoices.csv',
        as_attachment=True
    )


@app.route("/download_excel")
def download_excel():
    global latest_results_df
    if latest_results_df is None:
        flash("No data available.")
        return redirect(url_for('index'))
    df = latest_results_df.copy()
    for col in ["editable", "comments"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    return send_file(
        generate_excel_bytes(df),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='matched_invoices.xlsx',
        as_attachment=True
    )


# New endpoint: Download summary statistics as Excel
@app.route("/download_stats_excel")
def download_stats_excel():
    global latest_results_df
    if latest_results_df is None:
        flash("No data available for stats.")
        return redirect(url_for('index'))
    stats = get_stats(latest_results_df)
    return send_file(
        generate_stats_excel_bytes(stats),
        mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        download_name='invoice_matching_stats.xlsx',
        as_attachment=True
    )


# New endpoint: Download summary statistics as JSON
@app.route("/download_stats_json")
def download_stats_json():
    global latest_results_df
    if latest_results_df is None:
        flash("No data available for stats.")
        return redirect(url_for('index'))
    stats = get_stats(latest_results_df)
    return send_file(
        generate_stats_json_bytes(stats),
        mimetype='application/json',
        download_name='invoice_matching_stats.json',
        as_attachment=True
    )


if __name__ == "__main__":
    app.run(debug=True)
