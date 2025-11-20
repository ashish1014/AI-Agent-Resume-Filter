# 🧠 Resume Filter Tool

A Python-based **Resume Filtering Utility** that helps recruiters or hiring teams quickly evaluate candidate resumes based on **years of experience** and **technical skills**.  
It supports both **PDF** and **Word (.docx)** resumes and produces a neat, color-coded summary in the terminal and a CSV report.

---

## 🚀 Features

✅ Extracts text from both **PDF** and **DOCX** resumes  
✅ Detects total **years of experience** (e.g., "8+ years")  
✅ Identifies specific **technologies or skills** (e.g., "Java", "Python", ".NET")  
✅ Shows formatted results in the console using `tabulate`  
✅ Saves all results automatically in `eligibility_results.csv`  
✅ Optionally filters resumes based on experience, skill, or both  

---

## 🧱 Project Structure

```
resume_tool/
│
├── resume_filter_agent.py  # Main script
├── requirements.txt        # Dependencies
├── README.md               # Project documentation
├── .gitignore              # Ignored files (optional)
└── resumes/                # Folder to store uploaded resumes
```

---

## ⚙️ Requirements

- Python **3.8+**
- Required libraries:
  - pandas  
  - PyMuPDF  
  - python-docx  
  - tabulate  
  - tk (built-in with Python on Windows)

Install all dependencies in one go:
```bash
pip install -r requirements.txt
```

---

## ▶️ How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/resume-filter-tool.git
   cd resume-filter-tool
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the tool**
   ```bash
   python resume_filter_agent.py
   ```

4. **Select resumes**
   - A file dialog will open where you can select one or more `.pdf` or `.docx` resumes.

5. **Choose filtering criteria**
   - `1` → Filter by experience  
   - `2` → Filter by technology  
   - `1,2` → Filter by both  

6. **View results**
   - The tool will show a neat, formatted table in the terminal.
   - Results will also be saved in `eligibility_results.csv`.

---

## 🧮 Example Output

```
📄 Results saved to 'eligibility_results.csv'.

╒════════════════════╤═══════════════════╤════════════════╤══════════════╕
│ Candidate           │ Experience (yrs) │ Matched_Techs  │ Status       │
╞════════════════════╪═══════════════════╪════════════════╪══════════════╡
│ AshishSuman_Resume │ 8.0               │ Java           │ ✅ Eligible   │
╘════════════════════╧═══════════════════╧════════════════╧══════════════╛

✅ All resumes processed successfully!
```

---

## 📊 Output File

The script automatically generates:
```
eligibility_results.csv
```

Example content:
| Candidate           | Experience (yrs) | Matched_Techs | Status      |
|---------------------|------------------|----------------|-------------|
| AshishSuman_Resume | 8.0              | Java           | Eligible ✅ |

---

## 💡 Improvements (Optional)

You can enhance the tool by:
- Adding multiple skill filtering (e.g., "Java OR Python")
- Exporting data to Excel format with color-coded cells
- Adding GUI input for filters (Tkinter-based)
- Integrating with LinkedIn Resume Downloader

---

## 👨‍💻 Author

**Ashish Suman**  
📧 [ashish.jha752@gmail.com](mailto:ashish.jha752@gmail.com)  
🌐 [LinkedIn](https://www.linkedin.com/in/ashish-suman/)

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
