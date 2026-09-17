"""Parser tests using minimal synthetic fixtures that reproduce structural
edge cases found in the real Upwork footers. No verbatim real job
descriptions are committed here (see docs/data_quality_report.md); the
fixtures below only reproduce the *structure* that broke earlier
prototypes.
"""
import unittest

from src.data.parse import (
    clean_country,
    extract_footer,
    parse_category,
    parse_job,
    parse_location_requirement,
    parse_skills,
)


def footer_block(body: str) -> str:
    """Wrap a job body with a normal footer, for readability in tests."""
    return (
        f"{body}"
        "Budget: $500\n"
        "Posted On: February 17, 2024 09:09 UTC\n"
        "Category: Social Media Marketing\n"
        "Skills:Alpha,     Beta,     Gamma\n"
        "Skills:Alpha,     Beta,     Gamma\n"
        "click to apply"
    )


class TestFooterAnchoring(unittest.TestCase):
    def test_normal_footer(self):
        desc = footer_block("A normal job description. ")
        footer = extract_footer(desc)
        self.assertEqual(parse_category(footer), "Social Media Marketing")
        self.assertEqual(parse_skills(footer), ["Alpha", "Beta", "Gamma"])

    def test_category_word_in_body_does_not_confuse_parser(self):
        body = (
            "We need someone who understands our Category: pricing structure "
            "and can write about it. "
        )
        desc = footer_block(body)
        footer = extract_footer(desc)
        # The footer starts at the LAST "Posted On:", so the body's
        # "Category:" must never leak into the parsed category.
        self.assertEqual(parse_category(footer), "Social Media Marketing")

    def test_last_posted_on_wins_when_multiple_occur(self):
        # A body that itself contains a fake "Posted On:" marker (e.g. the
        # client pasted an old post). Only the LAST one is the real footer.
        body = (
            "This role replaces a contract we originally listed: "
            "Posted On: January 1, 2024 00:00 UTC Category: Fake Category "
            "Skills:Nope\n"
        )
        desc = footer_block(body)
        footer = extract_footer(desc)
        self.assertEqual(parse_category(footer), "Social Media Marketing")
        self.assertEqual(parse_skills(footer), ["Alpha", "Beta", "Gamma"])

    def test_no_footer_returns_empty(self):
        desc = "Just a plain description with no platform footer at all."
        self.assertEqual(extract_footer(desc), "")
        parsed = parse_job(desc)
        self.assertFalse(parsed.footer_found)
        self.assertIsNone(parsed.category)
        self.assertEqual(parsed.skills_raw, [])


class TestSkillParsing(unittest.TestCase):
    def test_duplicated_skills_block_uses_first_occurrence_only(self):
        desc = footer_block("Body text. ")
        footer = extract_footer(desc)
        self.assertEqual(parse_skills(footer), ["Alpha", "Beta", "Gamma"])

    def test_skill_with_internal_comma_is_not_split(self):
        # Real evidence: labels like "Religious, Charitable & Nonprofit"
        # contain a comma+single-space, distinct from the comma+multiple
        # spaces used as the actual separator between skills.
        footer = (
            "Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Sales & Marketing\n"
            "Skills:Religious, Charitable & Nonprofit,     Copywriting,     Writing\n"
            "click to apply"
        )
        self.assertEqual(
            parse_skills(footer),
            ["Religious, Charitable & Nonprofit", "Copywriting", "Writing"],
        )

    def test_no_skills_block(self):
        footer = (
            "Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Data Entry\n"
            "click to apply"
        )
        self.assertEqual(parse_skills(footer), [])

    def test_deduplicates_within_job_preserving_order(self):
        footer = (
            "Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Writing\n"
            "Skills:Python,     Java,     Python\n"
            "click to apply"
        )
        self.assertEqual(parse_skills(footer), ["Python", "Java"])


class TestMalformedFooter(unittest.TestCase):
    def test_incomplete_footer_missing_category_and_skills(self):
        footer = "Posted On: February 17, 2024 09:09 UTC\n"
        self.assertIsNone(parse_category(footer))
        self.assertEqual(parse_skills(footer), [])

    def test_missing_location_requirement_is_none(self):
        footer = (
            "Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Writing\n"
            "Skills:Python\n"
            "Country: United States\n"
            "click to apply"
        )
        self.assertIsNone(parse_location_requirement(footer))


class TestHtmlEntities(unittest.TestCase):
    def test_html_entities_decoded_in_footer_fields(self):
        desc = (
            "Body with an &amp; entity. "
            "Posted On: February 17, 2024 09:09 UTC\n"
            "Category: Sales &amp; Marketing\n"
            "Skills:B2B &amp; B2C,     Copywriting\n"
            "Country: C&amp;ote d&#39;Ivoire\n"
            "click to apply"
        )
        footer = extract_footer(desc)
        self.assertEqual(parse_category(footer), "Sales & Marketing")
        self.assertEqual(parse_skills(footer), ["B2B & B2C", "Copywriting"])

    def test_clean_country_decodes_entities(self):
        self.assertEqual(clean_country("C&amp;ote d&#39;Ivoire"), "C&ote d'Ivoire")
        self.assertIsNone(clean_country(None))


class TestParseJob(unittest.TestCase):
    def test_full_parse_job_structure(self):
        desc = footer_block("Some description body. ")
        parsed = parse_job(desc)
        self.assertTrue(parsed.footer_found)
        self.assertEqual(parsed.category, "Social Media Marketing")
        self.assertEqual(parsed.skills_raw, ["Alpha", "Beta", "Gamma"])
        self.assertIsNone(parsed.location_requirement)


if __name__ == "__main__":
    unittest.main()
