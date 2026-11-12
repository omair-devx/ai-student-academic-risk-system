// Presentation metadata for the 24 enrollment-time features.
//
// Labelled options follow the OFFICIAL UCI #697 variables table (fetched and
// cross-checked against data/raw so every code observed in the dataset is
// representable — one-hot blocks never see an unseen value from this form).
// The API model card remains the source of truth for WHICH fields exist;
// FIELD_META here only decides how each field is rendered.

export type FieldKind = "select" | "segmented" | "number";

export interface Option {
  label: string;
  value: number;
}

export interface FieldMeta {
  label: string;
  group: GroupId;
  kind: FieldKind;
  hint?: string;
  options?: Option[];
  step?: number;
}

export type GroupId =
  | "demographics"
  | "application"
  | "qualification"
  | "family"
  | "circumstances"
  | "macro";

export const GROUPS: { id: GroupId; title: string; blurb: string }[] = [
  {
    id: "demographics",
    title: "Student demographics",
    blurb: "Identity details recorded at enrollment.",
  },
  {
    id: "application",
    title: "Application path",
    blurb: "How and what the student applied for.",
  },
  {
    id: "qualification",
    title: "Prior education",
    blurb: "Previous qualification, grades and admission score.",
  },
  {
    id: "family",
    title: "Family background",
    blurb: "Parents' qualification and occupation codes (UCI codebook).",
  },
  {
    id: "circumstances",
    title: "Financial circumstances",
    blurb: "Status flags at enrollment — see the provenance note below.",
  },
  {
    id: "macro",
    title: "Macroeconomic context",
    blurb: "National indicators around the enrollment date.",
  },
];

const YES_NO: Option[] = [
  { label: "No", value: 0 },
  { label: "Yes", value: 1 },
];

const MARITAL: Option[] = [
  { label: "Single", value: 1 },
  { label: "Married", value: 2 },
  { label: "Widower", value: 3 },
  { label: "Divorced", value: 4 },
  { label: "Facto union (common-law)", value: 5 },
  { label: "Legally separated", value: 6 },
];

const APP_MODE: Option[] = [
  { label: "1st phase — general contingent", value: 1 },
  { label: "Ordinance No. 612/93", value: 2 },
  { label: "1st phase — special contingent (Azores)", value: 5 },
  { label: "Holder of another higher course", value: 7 },
  { label: "Ordinance No. 854-B/99", value: 10 },
  { label: "International student (bachelor)", value: 15 },
  { label: "1st phase — special contingent (Madeira)", value: 16 },
  { label: "2nd phase — general contingent", value: 17 },
  { label: "3rd phase — general contingent", value: 18 },
  { label: "Ordinance No. 533-A/99 b2 (different plan)", value: 26 },
  { label: "Ordinance No. 533-A/99 b3 (other institution)", value: 27 },
  { label: "Over 23 years old", value: 39 },
  { label: "Transfer", value: 42 },
  { label: "Change of course", value: 43 },
  { label: "Technological specialization diploma", value: 44 },
  { label: "Change of institution/course", value: 51 },
  { label: "Short cycle diploma holder", value: 53 },
  { label: "Change of institution/course (international)", value: 57 },
];

// Only codes observed in the data are offered (7 and 8 never occur).
const APP_ORDER: Option[] = [
  { label: "0 — first choice", value: 0 },
  { label: "1 — second choice", value: 1 },
  { label: "2 — third choice", value: 2 },
  { label: "3 — fourth choice", value: 3 },
  { label: "4 — fifth choice", value: 4 },
  { label: "5 — sixth choice", value: 5 },
  { label: "6 — seventh choice", value: 6 },
  { label: "9 — last choice", value: 9 },
];

const COURSE: Option[] = [
  { label: "33 — Biofuel Production Technologies", value: 33 },
  { label: "171 — Animation and Multimedia Design", value: 171 },
  { label: "8014 — Social Service (evening attendance)", value: 8014 },
  { label: "9003 — Agronomy", value: 9003 },
  { label: "9070 — Communication Design", value: 9070 },
  { label: "9085 — Veterinary Nursing", value: 9085 },
  { label: "9119 — Informatics Engineering", value: 9119 },
  { label: "9130 — Equinculture", value: 9130 },
  { label: "9147 — Management", value: 9147 },
  { label: "9238 — Social Service", value: 9238 },
  { label: "9254 — Tourism", value: 9254 },
  { label: "9500 — Nursing", value: 9500 },
  { label: "9556 — Oral Hygiene", value: 9556 },
  { label: "9670 — Advertising and Marketing Management", value: 9670 },
  { label: "9773 — Journalism and Communication", value: 9773 },
  { label: "9853 — Basic Education", value: 9853 },
  { label: "9991 — Management (evening attendance)", value: 9991 },
];

const PREVIOUS_QUAL: Option[] = [
  { label: "Secondary education", value: 1 },
  { label: "Bachelor's degree", value: 2 },
  { label: "Degree (licentiate)", value: 3 },
  { label: "Master's degree", value: 4 },
  { label: "Doctorate", value: 5 },
  { label: "Frequency of higher education", value: 6 },
  { label: "12th year — not completed", value: 9 },
  { label: "11th year — not completed", value: 10 },
  { label: "Other — 11th year schooling", value: 12 },
  { label: "10th year of schooling", value: 14 },
  { label: "10th year — not completed", value: 15 },
  { label: "Basic ed. 3rd cycle (9–11th year)", value: 19 },
  { label: "Basic ed. 2nd cycle (6–8th year)", value: 38 },
  { label: "Technological specialization course", value: 39 },
  { label: "Higher ed. — degree (1st cycle)", value: 40 },
  { label: "Professional higher technical course", value: 42 },
  { label: "Higher ed. — master (2nd cycle)", value: 43 },
];

const NACIONALITY: Option[] = [
  { label: "Portuguese", value: 1 },
  { label: "German", value: 2 },
  { label: "Spanish", value: 6 },
  { label: "Italian", value: 11 },
  { label: "Dutch", value: 13 },
  { label: "English", value: 14 },
  { label: "Lithuanian", value: 17 },
  { label: "Angolan", value: 21 },
  { label: "Cape Verdean", value: 22 },
  { label: "Guinean", value: 24 },
  { label: "Mozambican", value: 25 },
  { label: "Santomean", value: 26 },
  { label: "Turkish", value: 32 },
  { label: "Brazilian", value: 41 },
  { label: "Romanian", value: 62 },
  { label: "Moldova (Republic of)", value: 100 },
  { label: "Mexican", value: 101 },
  { label: "Ukrainian", value: 103 },
  { label: "Russian", value: 105 },
  { label: "Cuban", value: 108 },
  { label: "Colombian", value: 109 },
];

// Shared education-level dictionary (UCI), superset used by both parents.
// Mother's observed 29 codes / father's observed 34 codes, all covered here.
const PARENT_QUAL: Option[] = [
  { label: "Secondary education — 12th year or eq.", value: 1 },
  { label: "Bachelor's degree", value: 2 },
  { label: "Degree (licentiate)", value: 3 },
  { label: "Master's degree", value: 4 },
  { label: "Doctorate", value: 5 },
  { label: "Frequency of higher education", value: 6 },
  { label: "12th year — not completed", value: 9 },
  { label: "11th year — not completed", value: 10 },
  { label: "7th year (old)", value: 11 },
  { label: "Other — 11th year schooling", value: 12 },
  { label: "2nd year complementary high school", value: 13 },
  { label: "10th year of schooling", value: 14 },
  { label: "General commerce course", value: 18 },
  { label: "Basic ed. 3rd cycle (9–11th year)", value: 19 },
  { label: "Complementary high school course", value: 20 },
  { label: "Technical-professional course", value: 22 },
  { label: "Complementary high school — not concluded", value: 25 },
  { label: "7th year of schooling", value: 26 },
  { label: "2nd cycle general high school course", value: 27 },
  { label: "9th year — not completed", value: 29 },
  { label: "8th year of schooling", value: 30 },
  { label: "Administration and commerce course", value: 31 },
  { label: "Supplementary accounting and administration", value: 33 },
  { label: "Unknown", value: 34 },
  { label: "Can't read or write", value: 35 },
  { label: "Can read (below 4th year of schooling)", value: 36 },
  { label: "Basic ed. 1st cycle (1–4th year) or eq.", value: 37 },
  { label: "Basic ed. 2nd cycle (6–8th year) or eq.", value: 38 },
  { label: "Technological specialization course", value: 39 },
  { label: "Higher ed. — degree (1st cycle)", value: 40 },
  { label: "Specialized higher studies course", value: 41 },
  { label: "Professional higher technical course", value: 42 },
  { label: "Higher ed. — master (2nd cycle)", value: 43 },
  { label: "Higher ed. — doctorate (3rd cycle)", value: 44 },
];

export const FIELD_META: Record<string, FieldMeta> = {
  "Marital status": {
    label: "Marital status",
    group: "demographics",
    kind: "select",
    options: MARITAL,
  },
  Gender: {
    label: "Gender",
    group: "demographics",
    kind: "segmented",
    hint: "Dataset coding: 1 = male, 0 = female (docs/DATA.md).",
    options: [
      { label: "Female", value: 0 },
      { label: "Male", value: 1 },
    ],
  },
  "Age at enrollment": {
    label: "Age at enrollment",
    group: "demographics",
    kind: "number",
    hint: "years",
    step: 1,
  },
  Nacionality: {
    label: "Nationality",
    group: "demographics",
    kind: "select",
    hint: "UCI spells the column “Nacionality”.",
    options: NACIONALITY,
  },
  International: {
    label: "International student",
    group: "demographics",
    kind: "segmented",
    options: YES_NO,
  },
  "Application mode": {
    label: "Application mode",
    group: "application",
    kind: "select",
    hint: "Entry pathway code (UCI codebook).",
    options: APP_MODE,
  },
  "Application order": {
    label: "Application order",
    group: "application",
    kind: "select",
    hint: "Rank of this course in the student's choices.",
    options: APP_ORDER,
  },
  Course: {
    label: "Course",
    group: "application",
    kind: "select",
    options: COURSE,
  },
  "Daytime/evening attendance": {
    label: "Attendance schedule",
    group: "application",
    kind: "segmented",
    hint: "UCI: 1 = daytime, 0 = evening.",
    options: [
      { label: "Daytime", value: 1 },
      { label: "Evening", value: 0 },
    ],
  },
  "Previous qualification": {
    label: "Previous qualification",
    group: "qualification",
    kind: "select",
    hint: "Education level obtained before this program.",
    options: PREVIOUS_QUAL,
  },
  "Previous qualification (grade)": {
    label: "Previous qualification (grade)",
    group: "qualification",
    kind: "number",
    hint: "0–200 scale",
    step: 1,
  },
  "Admission grade": {
    label: "Admission grade",
    group: "qualification",
    kind: "number",
    hint: "0–200 scale",
    step: 1,
  },
  "Mother's qualification": {
    label: "Mother's qualification",
    group: "family",
    kind: "select",
    options: PARENT_QUAL,
  },
  "Father's qualification": {
    label: "Father's qualification",
    group: "family",
    kind: "select",
    options: PARENT_QUAL,
  },
  "Mother's occupation": {
    label: "Mother's occupation",
    group: "family",
    kind: "number",
    hint: "UCI occupation code (multi-level)",
    step: 1,
  },
  "Father's occupation": {
    label: "Father's occupation",
    group: "family",
    kind: "number",
    hint: "UCI occupation code (multi-level)",
    step: 1,
  },
  Displaced: {
    label: "Displaced student",
    group: "circumstances",
    kind: "segmented",
    options: YES_NO,
  },
  "Educational special needs": {
    label: "Educational special needs",
    group: "circumstances",
    kind: "segmented",
    options: YES_NO,
  },
  Debtor: {
    label: "Debtor (outstanding fees)",
    group: "circumstances",
    kind: "segmented",
    options: YES_NO,
  },
  "Tuition fees up to date": {
    label: "Tuition fees up to date",
    group: "circumstances",
    kind: "segmented",
    options: [
      { label: "Not up to date", value: 0 },
      { label: "Up to date", value: 1 },
    ],
  },
  "Scholarship holder": {
    label: "Scholarship holder",
    group: "circumstances",
    kind: "segmented",
    options: YES_NO,
  },
  "Unemployment rate": {
    label: "Unemployment rate",
    group: "macro",
    kind: "number",
    hint: "national indicator at enrollment (%)",
    step: 0.1,
  },
  "Inflation rate": {
    label: "Inflation rate",
    group: "macro",
    kind: "number",
    hint: "national indicator at enrollment (%)",
    step: 0.1,
  },
  GDP: {
    label: "GDP",
    group: "macro",
    kind: "number",
    hint: "growth indicator at enrollment (%)",
    step: 0.01,
  },
};

// Synthetic demonstration profile (required fields only, every value inside
// the observed training distribution). Clearly labelled as example data in
// the UI — no real student is ever represented.
export const EXAMPLE_STUDENT: Record<string, number> = {
  "Marital status": 1,
  Gender: 0,
  "Age at enrollment": 21,
  Nacionality: 1,
  International: 0,
  "Application mode": 1,
  "Application order": 2,
  Course: 9119,
  "Daytime/evening attendance": 1,
  "Previous qualification": 1,
  "Previous qualification (grade)": 130,
  "Admission grade": 125,
  "Mother's qualification": 1,
  "Father's qualification": 2,
  "Mother's occupation": 122,
  "Father's occupation": 4,
  Displaced: 0,
  "Educational special needs": 0,
  Debtor: 0,
  "Tuition fees up to date": 1,
  "Scholarship holder": 0,
  "Unemployment rate": 10.8,
  "Inflation rate": 0.6,
  GDP: 1.74,
};
