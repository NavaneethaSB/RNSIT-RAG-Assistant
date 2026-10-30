import unittest

from crawler import canonicalize_url, extract_content


class CrawlerTests(unittest.TestCase):
    def test_url_canonicalization_drops_tracking_but_keeps_content_query(self):
        result = canonicalize_url("http://rnsit.ac.in/program?utm_source=mail&year=2026#fees")
        self.assertEqual(result, "https://www.rnsit.ac.in/program/?year=2026")

    def test_external_and_non_content_hosts_are_rejected(self):
        self.assertIsNone(canonicalize_url("https://example.com/"))
        self.assertIsNone(canonicalize_url("https://subdomain.rnsit.ac.in/"))
        self.assertIsNone(canonicalize_url("https://www.rnsit.ac.in:9443/"))

    def test_extractor_preserves_title_canonical_url_and_removes_navigation(self):
        html = """
        <html><head><title>Admissions</title><link rel="canonical" href="/admissions/"></head>
        <body><div class="main_wrapper"><nav>Navigation Noise</nav><h1>Admissions</h1>
        <p>Official admission information for prospective students at RNSIT.</p>
        <footer>Footer noise</footer></div></body></html>
        """
        title, url, text, _ = extract_content(html, "https://www.rnsit.ac.in/admissions?utm_campaign=x")
        self.assertEqual(title, "Admissions")
        self.assertEqual(url, "https://www.rnsit.ac.in/admissions/")
        self.assertIn("Official admission information", text)
        self.assertNotIn("Navigation Noise", text)
        self.assertNotIn("Footer noise", text)


if __name__ == "__main__":
    unittest.main()
