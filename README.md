# Streamlit Number Place App

## **Overview**
This project is designed to solve and generate Number Place puzzles using mathematical optimization techniques. It offers an interactive user interface built with Streamlit, allowing users to input Number Place puzzles, generate unique puzzles, and view solutions.

---

## **Main Features**
1. **Solve Number Place Puzzles**: Input a puzzle and find the solution using mathematical optimization.
2. **Generate Unique Puzzles**: Automatically create Number Place puzzles with guaranteed unique solutions.
3. **Interactive UI**: A user-friendly interface powered by Streamlit for seamless interaction.
4. **Mathematical Optimization**: Utilize the PuLP library to solve constraints effectively.

---

## **Installation**
Follow these steps to set up the project on your local machine.

1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd number_place_solver
   ```

2. **Install Dependencies**:
   ```bash
   poetry install
   ```

3. **Run the Application**:
   ```bash
   poetry run streamlit run app/main.py
   ```

---

## **Project Structure**
The directory structure is organized as follows:

```
.
├── LICENSE               # License file
├── README.md             # Project overview (this file)
├── app/                  # Source code for the application
│   ├── main.py           # Entry point for the Streamlit app
│   └── utils.py          # Utility functions and logic
├── pyproject.toml        # Poetry configuration file
├── poetry.lock           # Dependency lock file
└── tests/                # Unit tests for the application
```

---

## **Dependencies**
The project uses the following major libraries:
- **Python 3.10**: The programming language version used.
- **Streamlit 1.40.2**: Framework for building interactive web apps.
- **PuLP 2.9.0**: Library for solving linear programming problems.
- **Pandas 2.2.3**: Data analysis and manipulation library.

---

## **License**
This project is licensed under the [Apache License 2.0](./LICENSE).

---

## **Contributing**
We welcome contributions! Follow these steps to get started:
1. Fork the repository.
2. Create a new branch for your feature:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add new feature"
   ```
4. Push your changes:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

If you'd like any additional details or adjustments to this `README.md`, feel free to ask!
