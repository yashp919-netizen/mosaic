---
purpose: Ask the LLM to produce a plain-English business description and quality flags for one source column
version: v1
agent: SCOUT
expected_output_schema: ColumnCharacterization
---

You are a senior data analyst reviewing a column from a CPG (Consumer Packaged Goods) product master dataset.

Given the column metadata below, produce:
1. A 1–2 sentence plain-English description of what this column likely contains and its business meaning.
2. A list of data quality flags that apply to this column.

Valid quality flags (use only these exact strings, include all that apply):
- "mixed_languages" — values appear in more than one language or script
- "abbreviated_codes" — values are short codes or abbreviations (e.g. PC-BW, SKU_CD)
- "free_text" — values are unstructured narrative text
- "high_cardinality_categorical" — string values with many distinct values that look categorical
- "numeric_as_string" — numbers stored as strings
- "date_format_variation" — dates in non-standard or inconsistent formats
- "high_null_rate" — more than 20% of values are null
- "test_data_present" — values contain TEST, XXX, DUMMY, or similar artifacts
- "encoding_issues" — values contain garbled characters (mojibake)
- "unit_ambiguity" — numeric values where the unit is unclear or inconsistent

Column metadata:
- Column name: {column_name}
- Inferred dtype: {inferred_dtype}
- Null rate: {null_rate_pct}%
- Detected language: {detected_language}
- Sample values: {value_samples}

Respond with a business description and the applicable quality flags only. Do not repeat the input metadata.
