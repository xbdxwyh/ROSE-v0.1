document.addEventListener("DOMContentLoaded", () => {
  const button = document.querySelector("[data-copy-target]");
  if (!button) return;

  button.addEventListener("click", async () => {
    const targetId = button.getAttribute("data-copy-target");
    const target = document.getElementById(targetId);
    if (!target) return;

    const text = target.innerText;
    const label = button.querySelector("span");

    try {
      await navigator.clipboard.writeText(text);
      if (label) label.textContent = "Copied";
      button.classList.add("copied");
      window.setTimeout(() => {
        if (label) label.textContent = "Copy";
        button.classList.remove("copied");
      }, 1600);
    } catch (error) {
      console.error("Could not copy BibTeX:", error);
    }
  });
});
