from playwright.sync_api import sync_playwright
import os

# A0 landscape in inches: 1189mm x 841mm -> /25.4
W_IN, H_IN = 1189/25.4, 841/25.4
url = "file://" + os.path.abspath("poster.html")

with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    # set a big viewport so the mm-sized poster lays out, then screenshot full page
    pg.set_viewport_size({"width": 1600, "height": 1132})
    pg.goto(url, wait_until="networkidle")
    pg.pdf(path="poster_A0.pdf", width=f"{W_IN}in", height=f"{H_IN}in",
           print_background=True, prefer_css_page_size=False)
    # PNG preview at a readable scale: render the .poster element
    el = pg.query_selector(".poster")
    el.screenshot(path="poster_preview.png")
    b.close()
print("done")
print("PDF size:", os.path.getsize("poster_A0.pdf")//1024, "KB")
print("PNG size:", os.path.getsize("poster_preview.png")//1024, "KB")
