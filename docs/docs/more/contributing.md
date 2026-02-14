(contributing)=
# Contributing 

🚀 Thank you for your interest in contributing to Ragrank! We welcome contributions from the community to help improve our product and make it even more robust. Below is a step-by-step guide on how you can contribute to Ragrank:

**1. Fork the Repository:**
   - Visit the Ragrank repository on GitHub: [Ragrank Repository](https://github.com/Auto-Playground/Ragrank).
   - Click on the "Fork" button in the top-right corner to create a copy of the repository in your GitHub account.

**2. Clone Your Fork:**
   - Open a terminal window.
   - Use the following command to clone your forked repository to your local machine:
     ```
     git clone https://github.com/{ your-username }/Ragrank.git
     ```

**3. Set Up Development Environment:**
   - Navigate to the cloned repository directory:
     ```
     cd Ragrank
     ```
   - Install the required dependencies by running:
     ```
     uv sync --group dev --group docs
     ```

**4. Make Changes:**
   - Create a new branch for your changes:
     ```
     git checkout -b my-feature
     ```
   - Implement your changes or additions to the codebase.

**5. Test Your Changes:**
   - Run tests to ensure your changes haven't introduced any regressions:
     ```
     make test
     ```

**6. Linting and formating:**
   - Format the code
     ```
     make format
     ```
   - Check the linting
     ```
     make lint
     ```
**7. Commit Your Changes:**
   - Once you're satisfied with your changes, commit them:
     ```
     git add .
     git commit -m "Add your descriptive commit message here"
     ```

**8. Push Changes to Your Fork:**
   - Push your changes to your forked repository:
     ```
     git push origin my-feature
     ```

**9. Create a Pull Request:**
   - Go to your forked repository on GitHub.
   - Click on the "Compare & pull request" button next to your branch.
   - Fill out the pull request form with a descriptive title and details of your changes.
   - Click on the "Create pull request" button to submit your contribution.

**10. Collaborate and Iterate:**
   - Engage with reviewers and address any feedback or requests for changes.
   - Iterate on your code until it meets the project's standards and requirements.

**11. Stay Updated:**
   - Keep an eye on the pull request for any updates or requests from maintainers.
   - Stay engaged with the Ragrank community and contribute to discussions and future development efforts.

🙌 Thank you for your contribution to Ragrank! Your efforts help make our product better for everyone. If you have any questions or need assistance, don't hesitate to reach out to us through GitHub or our community channels. Happy coding!