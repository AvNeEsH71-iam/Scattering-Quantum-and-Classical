const revealEls = document.querySelectorAll('.reveal');

const observer = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.classList.add('is-visible');
      }
    });
  },
  { threshold: 0.18 }
);

revealEls.forEach((el, idx) => {
  el.style.transitionDelay = `${Math.min(idx * 90, 380)}ms`;
  observer.observe(el);
});

document.getElementById('year').textContent = new Date().getFullYear();
