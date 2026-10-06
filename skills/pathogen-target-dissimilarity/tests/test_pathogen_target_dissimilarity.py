"""Tests for pathogen-target-dissimilarity.

Written before the implementation, per the repo's red/green rule.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from homology_search import (
    HomologyHit,
    best_hit_per_query,
    coverage_from_spans,
    merge_spans,
    parse_tabular_hits,
    read_fasta,
    read_fasta_with_headers,
    search_pyhmmer,
)
from pathogen_target_dissimilarity import (
    DEMO_DIR,
    annotations_from_fasta,
    build_demo_inputs,
    merge_annotations,
    read_annotations,
    run,
)
from target_scoring import (
    CONSERVATION_ALL_POINTS,
    ESSENTIAL_POINTS,
    ESSENTIAL_UNKNOWN_POINTS,
    SAFETY_PENALTY,
    decide,
    score_conservation,
    score_essentiality,
    score_selectivity,
    score_target,
)


# ---------------------------------------------------------------------------
# FASTA and hit parsing
# ---------------------------------------------------------------------------

class TestReadFasta:
    def test_reads_records(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text(">a desc here\nMKV\nLLA\n>b\nMMM\n")
        records = read_fasta(path)
        assert list(records) == [("a", "MKVLLA"), ("b", "MMM")]

    def test_description_is_stripped_from_id(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text(">sp|P12345|NAME_HUMAN some protein\nMKV\n")
        assert read_fasta(path)[0][0] == "sp|P12345|NAME_HUMAN"

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text("")
        assert read_fasta(path) == []

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(SystemExit):
            read_fasta(tmp_path / "nope.faa")


class TestParseTabularHits:
    """BLAST/DIAMOND outfmt 6: qseqid sseqid pident length ... evalue bitscore."""

    def test_parses_standard_columns(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(
            "q1\ts1\t38.5\t210\t100\t2\t1\t210\t5\t215\t1e-40\t150.0\n"
            "q2\ts2\t72.1\t300\t80\t1\t1\t300\t1\t300\t1e-90\t320.0\n"
        )
        hits = parse_tabular_hits(path)
        assert len(hits) == 2
        assert hits[0].query_id == "q1"
        assert hits[0].identity_pct == 38.5
        assert hits[0].evalue == 1e-40

    def test_skips_comment_lines(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text("# comment\nq1\ts1\t38.5\t210\t100\t2\t1\t210\t5\t215\t1e-40\t150.0\n")
        assert len(parse_tabular_hits(path)) == 1

    def test_malformed_line_is_rejected_not_guessed(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text("q1\ts1\t38.5\n")
        with pytest.raises(SystemExit):
            parse_tabular_hits(path)


class TestSpanCoverage:
    """Coverage is the whole aligned fraction of the query, not one segment of it."""

    def test_single_span(self):
        assert merge_spans([(1, 10)]) == 10

    def test_disjoint_spans_add(self):
        assert merge_spans([(1, 10), (21, 30)]) == 20

    def test_overlapping_spans_counted_once(self):
        assert merge_spans([(1, 10), (5, 15)]) == 15

    def test_adjacent_spans_merge(self):
        assert merge_spans([(1, 10), (11, 20)]) == 20

    def test_reversed_pair_is_normalised(self):
        assert merge_spans([(10, 1)]) == 10

    def test_empty(self):
        assert merge_spans([]) == 0

    def test_coverage_sums_every_segment(self):
        # Four domains of 100 each across a 1000-residue query: 40%, not 10%.
        spans = [(1, 100), (201, 300), (401, 500), (601, 700)]
        assert coverage_from_spans(spans, 1000) == 40.0

    def test_coverage_needs_positive_length(self):
        assert coverage_from_spans([(1, 10)], 0) is None


class TestMultiHspCoverage:
    """outfmt 6 writes one row per HSP; a true orthologue often spans several."""

    def test_hsps_of_one_pair_are_merged(self, tmp_path):
        path = tmp_path / "hits.tsv"
        # Two HSPs, 300 residues each, on a 1000-residue query: one hit at 60%.
        path.write_text(
            "q1\ts1\t40.0\t300\t0\t0\t1\t300\t1\t300\t1e-90\t320.0\t1000\n"
            "q1\ts1\t30.0\t300\t0\t0\t501\t800\t501\t800\t1e-40\t150.0\t1000\n"
        )
        hits = parse_tabular_hits(path)
        assert len(hits) == 1
        assert hits[0].coverage_pct == 60.0
        assert hits[0].evalue == 1e-90
        assert hits[0].identity_pct == 35.0

    def test_distinct_subjects_stay_separate(self, tmp_path):
        path = tmp_path / "hits.tsv"
        path.write_text(
            "q1\ts1\t40.0\t100\t0\t0\t1\t100\t1\t100\t1e-20\t90.0\t500\n"
            "q1\ts2\t40.0\t100\t0\t0\t1\t100\t1\t100\t1e-20\t90.0\t500\n"
        )
        assert len(parse_tabular_hits(path)) == 2

    def test_multi_hsp_orthologue_survives_the_coverage_filter(self, tmp_path):
        """The regression: four 12% HSPs are a 48% orthologue, not four coincidences."""
        from homology_search import filter_by_coverage

        path = tmp_path / "hits.tsv"
        rows = "".join(
            f"q1\ts1\t45.0\t120\t0\t0\t{s}\t{s+119}\t{s}\t{s+119}\t1e-92\t300.0\t1000\n"
            for s in (1, 201, 401, 601)
        )
        path.write_text(rows)
        hits = parse_tabular_hits(path)
        assert hits[0].coverage_pct == 48.0
        assert filter_by_coverage(hits, 40.0) == hits


class TestBestHitPerQuery:
    def test_lowest_evalue_wins(self):
        hits = [
            HomologyHit("q1", "s1", 30.0, 50.0, 1e-10),
            HomologyHit("q1", "s2", 25.0, 50.0, 1e-40),
        ]
        assert best_hit_per_query(hits)["q1"].subject_id == "s2"

    def test_identity_breaks_evalue_tie(self):
        hits = [
            HomologyHit("q1", "s1", 30.0, 50.0, 1e-20),
            HomologyHit("q1", "s2", 55.0, 50.0, 1e-20),
        ]
        assert best_hit_per_query(hits)["q1"].subject_id == "s2"

    def test_queries_kept_separate(self):
        hits = [
            HomologyHit("q1", "s1", 30.0, 50.0, 1e-20),
            HomologyHit("q2", "s2", 40.0, 50.0, 1e-20),
        ]
        assert set(best_hit_per_query(hits)) == {"q1", "q2"}

    def test_empty_input(self):
        assert best_hit_per_query([]) == {}


class TestSearchPyhmmer:
    def test_finds_identical_sequence_at_high_identity(self):
        seq = "MKVLLATTLLASSAWAQEPTVAQIQELAKRHNVDPQLLLALLQAEQGGRQFDANGKVLNS"
        hits = search_pyhmmer([("q1", seq)], [("s1", seq)])
        assert hits
        assert hits[0].identity_pct > 95.0

    def test_unrelated_sequences_give_no_hit(self):
        query = "MKVLLATTLLASSAWAQEPTVAQIQELAKRHNVDPQLLLALLQAEQGGRQFDANGKVLNS"
        target = "WWWWCWWCWWCWWWCWWCWWWCWWCWWWCWWCWWWCWWCWWWCWWCWWWCWWCWWWCWWC"
        assert search_pyhmmer([("q1", query)], [("s1", target)]) == []

    def test_coverage_is_reported(self):
        seq = "MKVLLATTLLASSAWAQEPTVAQIQELAKRHNVDPQLLLALLQAEQGGRQFDANGKVLNS"
        hits = search_pyhmmer([("q1", seq)], [("s1", seq)])
        assert 0.0 < hits[0].coverage_pct <= 100.0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoreSelectivity:
    def test_no_host_hit_scores_highest(self):
        points, reason = score_selectivity(None)
        assert points == 40
        assert "no detectable" in reason.lower()

    @pytest.mark.parametrize("identity,expected", [(10.0, 32), (24.9, 32), (25.0, 20), (39.9, 20), (40.0, 8), (59.9, 8), (60.0, 0), (95.0, 0)])
    def test_bands(self, identity, expected):
        points, _ = score_selectivity(HomologyHit("q", "s", identity, 80.0, 1e-30))
        assert points == expected


class TestScoreEssentiality:
    def test_essential_with_source(self):
        points, reason = score_essentiality("essential", "GRACE")
        assert points == ESSENTIAL_POINTS
        assert "GRACE" in reason

    def test_essential_without_source_is_downgraded(self):
        points, _ = score_essentiality("essential", "")
        assert points == ESSENTIAL_UNKNOWN_POINTS

    def test_unknown(self):
        assert score_essentiality("unknown", "")[0] == ESSENTIAL_UNKNOWN_POINTS

    def test_blank_is_unknown(self):
        assert score_essentiality("", "")[0] == ESSENTIAL_UNKNOWN_POINTS

    def test_non_essential_scores_zero(self):
        assert score_essentiality("non-essential", "GRACE")[0] == 0

    def test_unrecognised_call_is_not_guessed(self):
        points, reason = score_essentiality("probably essential", "GRACE")
        assert points == ESSENTIAL_UNKNOWN_POINTS
        assert "unrecognised" in reason.lower()


class TestScoreConservation:
    def test_present_in_all_panel_species(self):
        points, _ = score_conservation(3, 3)
        assert points == CONSERVATION_ALL_POINTS

    def test_present_in_half(self):
        points, _ = score_conservation(2, 4)
        assert points == 13

    def test_present_in_few(self):
        assert score_conservation(1, 4)[0] == 0

    def test_no_panel_supplied_scores_zero_not_full(self):
        points, reason = score_conservation(0, 0)
        assert points == 0
        assert "no panel" in reason.lower()


class TestScoreTarget:
    def _row(self, **kwargs):
        base = {
            "protein_id": "P1",
            "gene_name": "ERG11",
            "description": "lanosterol 14-alpha-demethylase",
            "host_hit": None,
            "essentiality": "essential",
            "essentiality_source": "GRACE",
            "panel_present": 3,
            "panel_total": 3,
        }
        base.update(kwargs)
        return base

    def test_best_case_scores_100(self):
        verdict = score_target(**self._row())
        assert verdict.score == 100
        assert verdict.decision == "pursue"

    def test_close_host_orthologue_penalised(self):
        verdict = score_target(**self._row(host_hit=HomologyHit("P1", "H1", 72.0, 90.0, 1e-80)))
        assert any("selectivity penalty" in r for r in verdict.reasons)
        assert verdict.score == 100 - 40 - SAFETY_PENALTY

    def test_score_never_negative(self):
        verdict = score_target(**self._row(
            host_hit=HomologyHit("P1", "H1", 95.0, 99.0, 1e-99),
            essentiality="non-essential",
            panel_present=0,
            panel_total=3,
        ))
        assert verdict.score == 0

    def test_missing_annotations_are_recorded(self):
        verdict = score_target(**self._row(essentiality="", essentiality_source="", panel_total=0, panel_present=0))
        assert "essentiality" in verdict.missing_annotations
        assert "conservation_panel" in verdict.missing_annotations

    def test_host_search_not_run_is_distinct_from_no_hit(self):
        verdict = score_target(**self._row(host_hit=None, host_searched=False))
        assert "host_homology" in verdict.missing_annotations
        assert verdict.score < 100

    def test_reasons_are_populated(self):
        verdict = score_target(**self._row())
        assert len(verdict.reasons) >= 3


class TestDecide:
    @pytest.mark.parametrize("score,expected", [(70, "pursue"), (69, "hold"), (45, "hold"), (44, "drop"), (0, "drop")])
    def test_boundaries(self, score, expected):
        assert decide(score) == expected


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def demo_output(tmp_path_factory):
    """Run the skill once on the built-in demo inputs."""
    output = tmp_path_factory.mktemp("ptd")
    inputs = build_demo_inputs()
    run(output_dir=output, argv=["--demo"], **inputs)
    return output


class TestOutputContract:
    @pytest.mark.parametrize("relative", [
        "report.md",
        "result.json",
        "tables/target_scores.csv",
        "tables/host_hits.csv",
        "reproducibility/commands.sh",
        "reproducibility/environment.yml",
        "reproducibility/checksums.sha256",
    ])
    def test_artifact_written(self, demo_output, relative):
        assert (demo_output / relative).exists()

    def test_report_carries_disclaimer(self, demo_output):
        assert "Not a medical device" in (demo_output / "report.md").read_text()

    def test_result_json_valid_and_sorted(self, demo_output):
        payload = json.loads((demo_output / "result.json").read_text())
        scores = [t["score"] for t in payload["targets"]]
        assert scores == sorted(scores, reverse=True)

    def test_every_target_has_a_decision(self, demo_output):
        payload = json.loads((demo_output / "result.json").read_text())
        assert all(t["decision"] in {"pursue", "hold", "drop"} for t in payload["targets"])

    def test_demo_files_exist_on_disk(self):
        assert (DEMO_DIR / "demo_pathogen.faa").exists()
        assert (DEMO_DIR / "demo_host.faa").exists()
        assert (DEMO_DIR / "demo_essentiality.csv").exists()

    def test_run_is_deterministic(self, tmp_path):
        inputs = build_demo_inputs()
        first = tmp_path / "a"
        second = tmp_path / "b"
        run(output_dir=first, argv=["--demo"], **inputs)
        run(output_dir=second, argv=["--demo"], **inputs)
        a = json.loads((first / "result.json").read_text())["targets"]
        b = json.loads((second / "result.json").read_text())["targets"]
        assert [t["score"] for t in a] == [t["score"] for t in b]

    def test_conserved_housekeeping_target_is_dropped(self, demo_output):
        """The demo includes a protein with a close host orthologue; it must not rank top."""
        payload = json.loads((demo_output / "result.json").read_text())
        by_gene = {t["gene_name"]: t for t in payload["targets"]}
        assert by_gene["ACT1"]["decision"] == "drop"

    def test_fungal_specific_target_is_pursued(self, demo_output):
        payload = json.loads((demo_output / "result.json").read_text())
        by_gene = {t["gene_name"]: t for t in payload["targets"]}
        assert by_gene["FKS1"]["decision"] == "pursue"

    def test_best_case_demo_target_scores_full_marks(self, demo_output):
        """FKS1 has no host homologue, a sourced essentiality call and full panel presence.

        This caught a demo-data defect: an unquoted comma in the FKS1
        description shifted every later column, silently costing it its
        essentiality call.
        """
        payload = json.loads((demo_output / "result.json").read_text())
        by_gene = {t["gene_name"]: t for t in payload["targets"]}
        assert by_gene["FKS1"]["score"] == 100
        assert by_gene["FKS1"]["missing_annotations"] == []

    def test_descriptions_with_commas_survive_parsing(self, demo_output):
        payload = json.loads((demo_output / "result.json").read_text())
        by_gene = {t["gene_name"]: t for t in payload["targets"]}
        assert by_gene["FKS1"]["description"] == "beta-1,3-glucan synthase catalytic subunit"


# ---------------------------------------------------------------------------
# UniProt header parsing (added for real-proteome runs)
# ---------------------------------------------------------------------------

from homology_search import parse_uniprot_header  # noqa: E402


class TestParseUniprotHeader:
    def test_swissprot_header_with_gene_name(self):
        header = "sp|P0CY33|ERG11_CANAL Lanosterol 14-alpha demethylase OS=Candida albicans OX=237561 GN=ERG11 PE=1 SV=1"
        parsed = parse_uniprot_header(header)
        assert parsed["gene_name"] == "ERG11"
        assert parsed["description"] == "Lanosterol 14-alpha demethylase"

    def test_trembl_header(self):
        header = "tr|A0A1D8PI61|A0A1D8PI61_CANAL Msu1p OS=Candida albicans OX=237561 GN=MSU1 PE=4 SV=1"
        parsed = parse_uniprot_header(header)
        assert parsed["gene_name"] == "MSU1"
        assert parsed["description"] == "Msu1p"

    def test_header_without_gene_name_falls_back_to_entry_name(self):
        header = "sp|Q5A933|Q5A933_CANAL Uncharacterized protein OS=Candida albicans OX=237561 PE=4 SV=1"
        parsed = parse_uniprot_header(header)
        assert parsed["gene_name"] == "Q5A933_CANAL"
        assert parsed["description"] == "Uncharacterized protein"

    def test_non_uniprot_header_is_left_alone(self):
        parsed = parse_uniprot_header("CAL_FKS1 beta-1,3-glucan synthase")
        assert parsed["gene_name"] == ""
        assert parsed["description"] == "beta-1,3-glucan synthase"

    def test_bare_identifier(self):
        parsed = parse_uniprot_header("CAL_FKS1")
        assert parsed["gene_name"] == ""
        assert parsed["description"] == ""

    def test_description_stops_before_os_field(self):
        header = "sp|P0CY33|ERG11_CANAL Cytochrome P450 51 OS=Candida albicans OX=237561 GN=ERG11"
        assert "OS=" not in parse_uniprot_header(header)["description"]


class TestReadFastaWithHeaders:
    def test_headers_are_retained(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text(">sp|P0CY33|ERG11_CANAL Lanosterol 14-alpha demethylase OS=Candida albicans GN=ERG11\nMKV\n")
        records = read_fasta_with_headers(path)
        assert records[0][0] == "sp|P0CY33|ERG11_CANAL"
        assert records[0][2] == "sp|P0CY33|ERG11_CANAL Lanosterol 14-alpha demethylase OS=Candida albicans GN=ERG11"

    def test_identifier_matches_read_fasta(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text(">sp|P0CY33|ERG11_CANAL desc here\nMKV\n")
        assert read_fasta(path)[0][0] == read_fasta_with_headers(path)[0][0]


class TestAnnotationsFromHeaders:
    def test_builds_annotations_when_no_csv_given(self, tmp_path):
        path = tmp_path / "p.faa"
        path.write_text(
            ">sp|P0CY33|ERG11_CANAL Lanosterol 14-alpha demethylase OS=Candida albicans GN=ERG11\nMKV\n"
            ">sp|Q5A933|Q5A933_CANAL Uncharacterized protein OS=Candida albicans\nMMM\n"
        )
        annotations = annotations_from_fasta(path)
        assert annotations["sp|P0CY33|ERG11_CANAL"]["gene_name"] == "ERG11"
        assert annotations["sp|Q5A933|Q5A933_CANAL"]["essentiality"] == ""

    def test_supplied_csv_wins_over_headers(self, tmp_path):
        fasta = tmp_path / "p.faa"
        fasta.write_text(">sp|P0CY33|ERG11_CANAL Lanosterol demethylase GN=ERG11\nMKV\n")
        csv_path = tmp_path / "a.csv"
        csv_path.write_text(
            "protein_id,gene_name,description,essentiality,essentiality_source\n"
            "sp|P0CY33|ERG11_CANAL,ERG11,curated description,essential,GRACE\n"
        )
        merged = merge_annotations(annotations_from_fasta(fasta), read_annotations(csv_path))
        assert merged["sp|P0CY33|ERG11_CANAL"]["description"] == "curated description"
        assert merged["sp|P0CY33|ERG11_CANAL"]["essentiality"] == "essential"

    def test_header_fills_gaps_the_csv_leaves(self, tmp_path):
        fasta = tmp_path / "p.faa"
        fasta.write_text(">sp|P0CY33|ERG11_CANAL Lanosterol demethylase GN=ERG11\nMKV\n")
        csv_path = tmp_path / "a.csv"
        csv_path.write_text(
            "protein_id,gene_name,description,essentiality,essentiality_source\n"
            "sp|P0CY33|ERG11_CANAL,,,essential,GRACE\n"
        )
        merged = merge_annotations(annotations_from_fasta(fasta), read_annotations(csv_path))
        assert merged["sp|P0CY33|ERG11_CANAL"]["gene_name"] == "ERG11"
        assert merged["sp|P0CY33|ERG11_CANAL"]["essentiality_source"] == "GRACE"


# ---------------------------------------------------------------------------
# Orphan gate (added after the first real-proteome run)
# ---------------------------------------------------------------------------

class TestOrphanGate:
    """No host homologue only counts when the protein exists elsewhere in the clade.

    The first real run showed 63% of random C. albicans proteins had no human
    hit, against 51% of well-characterised ones: absence tracks being
    uncharacterised, not being a good target. 60 proteins had no host hit and
    no panel homologue and still scored 47.
    """

    def _row(self, **kwargs):
        base = {
            "protein_id": "P1", "gene_name": "X", "description": "d",
            "host_hit": None, "essentiality": "", "essentiality_source": "",
            "panel_present": 2, "panel_total": 2,
        }
        base.update(kwargs)
        return base

    def test_absent_from_whole_panel_forfeits_the_no_host_bonus(self):
        verdict = score_target(**self._row(panel_present=0, panel_total=2))
        assert "orphan_candidate" in verdict.flags
        assert verdict.score == 7

    def test_present_in_one_panel_species_keeps_the_bonus(self):
        verdict = score_target(**self._row(panel_present=1, panel_total=2))
        assert "orphan_candidate" not in verdict.flags
        assert verdict.score == 40 + 7 + 13

    def test_no_panel_supplied_keeps_bonus_but_flags_it(self):
        verdict = score_target(**self._row(panel_present=0, panel_total=0))
        assert "orphan_candidate" not in verdict.flags
        assert "conservation_panel" in verdict.missing_annotations
        assert verdict.score == 40 + 7

    def test_gate_does_not_apply_when_a_host_homologue_exists(self):
        hit = HomologyHit("P1", "H1", 35.0, 80.0, 1e-30)
        verdict = score_target(**self._row(host_hit=hit, panel_present=0, panel_total=2))
        assert "orphan_candidate" not in verdict.flags
        assert verdict.score == 20 + 7

    def test_flags_appear_in_the_scored_table(self, tmp_path):
        inputs = build_demo_inputs()
        run(output_dir=tmp_path, argv=["--demo"], **inputs)
        header = (tmp_path / "tables" / "target_scores.csv").read_text().splitlines()[0]
        assert "flags" in header
