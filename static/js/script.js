
document.addEventListener("DOMContentLoaded", function () {

    // Mobile navigation
    const menuToggle = document.getElementById("menuToggle");
    const navLinks = document.getElementById("navLinks");

    if (menuToggle && navLinks) {

        menuToggle.addEventListener("click", function () {

            const isOpen = navLinks.classList.toggle("show");

            menuToggle.setAttribute(
                "aria-expanded",
                String(isOpen)
            );

            menuToggle.textContent = isOpen ? "✕" : "☰";

        });

        navLinks.querySelectorAll("a").forEach(function (link) {

            link.addEventListener("click", function () {

                navLinks.classList.remove("show");

                menuToggle.textContent = "☰";

                menuToggle.setAttribute(
                    "aria-expanded",
                    "false"
                );

            });

        });

        document.addEventListener("click", function (event) {

            if (
                !navLinks.contains(event.target) &&
                !menuToggle.contains(event.target)
            ) {

                navLinks.classList.remove("show");

                menuToggle.textContent = "☰";

                menuToggle.setAttribute(
                    "aria-expanded",
                    "false"
                );

            }

        });

    }

    // Prediction form validation
    const predictionForm = document.getElementById("predictionForm");

    if (predictionForm) {

        predictionForm.addEventListener("submit", function (event) {

            const requiredFields = predictionForm.querySelectorAll(
                "[required]"
            );

            let valid = true;

            requiredFields.forEach(function (field) {

                if (!field.value.trim()) {

                    valid = false;

                    field.style.borderColor = "#ff7777";

                } else {

                    field.style.borderColor = "";

                }

            });

            if (!valid) {

                event.preventDefault();

                alert("Please complete all required vehicle details.");

            }

        });

    }

});
