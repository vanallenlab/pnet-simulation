# Unit tests for pnet_loader.py classes and functions

import os
import unittest

import pandas as pd
import torch

from pnet import pnet_loader, report_and_eval


class TestGenerateFeatureNames(unittest.TestCase):
    def test_generate_feature_names(self):
        genes = ["GeneA", "GeneB"]
        modalities = ["mut", "cnv"]
        embedding_length = 2

        expected = [
            "GeneA_mut",
            "GeneA_cnv",
            "GeneA_embedding1",
            "GeneA_embedding2",
            "GeneB_mut",
            "GeneB_cnv",
            "GeneB_embedding1",
            "GeneB_embedding2",
        ]

        result = pnet_loader.generate_feature_names(genes, modalities, embedding_length)
        self.assertEqual(result, expected)


class TestPnetDatasetClass(unittest.TestCase):
    def setUp(self):
        """Set up the test case with example genetic data and embeddings."""
        # Example 1: Simple case with one overlapping gene
        self.indices = ["Sample1", "Sample2"]
        self.target = pd.Series([1, 0], index=self.indices)

        self.genetic_data = {
            "mut": pd.DataFrame({"GeneA": [1, 0], "GeneB": [0, 1]}, index=self.indices),
            "cnv": pd.DataFrame({"GeneA": [0, 1], "GeneC": [1, 0]}, index=self.indices),
        }
        self.gene_embeddings = pd.DataFrame({"Embedding1": [0.5, 0.2]}, index=["GeneA", "GeneB"])

        # Example 2: More complex case with 2 overlapping genes, 3 modalities
        self.complex_indices = ["Sample1", "Sample2"]
        self.complex_target = pd.Series([1, 0], index=self.indices)
        self.complex_genetic_data = {
            "mut": pd.DataFrame({"GeneA": [1, 0], "GeneB": [0, 1], "GeneD": [1, 1]}, index=self.complex_indices),
            "del": pd.DataFrame({"GeneA": [0, 1], "GeneC": [1, 0], "GeneD": [1, 1]}, index=self.complex_indices),
            "amp": pd.DataFrame({"GeneA": [1, 1], "GeneD": [0, 1]}, index=self.complex_indices),
        }
        self.complex_gene_embeddings = pd.DataFrame({"Embedding1": [0.5, 0.2, 0.8]}, index=["GeneA", "GeneB", "GeneD"])

    def test_get_genes(self):
        """Test the get_genes method for overlapping genes."""
        dataset = pnet_loader.PnetDataset(genetic_data=self.genetic_data, target=self.target, indicies=self.indices)
        dataset.gene_embeddings = self.gene_embeddings
        expected_genes = ["GeneA"]
        self.assertEqual(sorted(dataset.get_genes()), sorted(expected_genes))

    def test_generate_input_mask(self):
        print("Testing complex case")
        complex_dataset = pnet_loader.PnetDataset(
            genetic_data=self.complex_genetic_data, target=self.complex_target, indicies=self.complex_indices
        )
        complex_dataset.gene_embeddings = self.complex_gene_embeddings

        mask_complex = complex_dataset.generate_input_mask()
        print(mask_complex)

        # Print the complex case mask for inspection
        expected_row_order = pnet_loader.generate_feature_names(complex_dataset.genes, complex_dataset.modalities)
        print("Complex case input mask:")
        print(pd.DataFrame(mask_complex, index=expected_row_order, columns=complex_dataset.genes))

        # Expected DataFrame for explicit equality check
        genes = ["GeneA", "GeneD"]
        modalities = ["mut", "del", "amp"]
        rows = [f"{gene}_{modality}" for gene in genes for modality in modalities]
        expected_mask = pd.DataFrame(0, index=rows, columns=genes)
        for gene in genes:
            for modality in modalities:
                expected_mask.loc[f"{gene}_{modality}", gene] = 1
        print("Expected input mask for complex case")

        self.assertTrue(
            (mask_complex == expected_mask.values).all(),
            "The input mask for the complex case does not match what we expect.",
        )

    @unittest.skip(
        "Skipping the test to compare Marc G vs my input mask generation for now b/c I realized that the difference is just the order of rows and how row names are made."
    )
    def test_generate_input_mask_easy(self):
        """Test that generate_input_mask and generate_input_mask_marc_version produce equivalent outputs."""
        dataset = pnet_loader.PnetDataset(genetic_data=self.genetic_data, target=self.target, indicies=self.indices)

        # Calculate masks
        mask1 = dataset.generate_input_mask()
        mask2 = dataset.generate_input_mask_marc_version()

        # Assert that the two masks are equal
        self.assertTrue((mask1 == mask2).all(), "The input masks are not equivalent.")

        # Optionally, print the mask for verification
        print("Generated Input Mask:")
        print(
            pd.DataFrame(
                mask1,
                index=pnet_loader.generate_feature_names(dataset.genes, dataset.modalities),
                columns=dataset.genes,
            )
        )

    @unittest.skip(
        "Skipping the test to compare Marc G vs my input mask generation for now b/c I realized that the difference is just in how row names are made. Does reaffirm my desire to control the row order, though. It's too easy to make a silent error."
    )
    def test_generate_input_mask_versions(self):
        """Test that generate_input_mask and generate_input_mask_marc_version produce equivalent outputs."""
        print("Testing simple case")
        dataset = pnet_loader.PnetDataset(genetic_data=self.genetic_data, target=self.target, indicies=self.indices)

        mask1 = dataset.generate_input_mask()
        mask2 = dataset.generate_input_mask_marc_version()

        self.assertTrue((mask1 == mask2).all(), "The input masks for the simple case are not equivalent.")

        print("Testing complex case")
        complex_dataset = pnet_loader.PnetDataset(
            genetic_data=self.complex_genetic_data, target=self.complex_target, indicies=self.complex_indices
        )
        complex_dataset.gene_embeddings = self.complex_gene_embeddings
        mask1_complex = complex_dataset.generate_input_mask()
        print(mask1_complex)
        mask2_complex = complex_dataset.generate_input_mask_marc_version()  # currently, this function produces mask in different row order. This type of silent error is why I really want to make row order controlled.
        print(mask2_complex)

        # Print the complex case mask for inspection
        expected_row_order = pnet_loader.generate_feature_names(complex_dataset.genes, complex_dataset.modalities)
        print("Complex Case Input Mask:")
        print(pd.DataFrame(mask1_complex, index=expected_row_order, columns=complex_dataset.genes))

        self.assertTrue(
            (mask1_complex == mask2_complex).all(), "The input masks for the complex case are not equivalent."
        )


class Test_report_and_eval(unittest.TestCase):
    def test_save_predictions_and_probs(self):
        # Create a temporary directory
        save_dir = "tmp_test_results"
        os.makedirs(save_dir, exist_ok=True)

        # Dummy data for testing
        who = "test"
        y_true = [0, 1, 1, 0]
        y_preds = [0, 1, 0, 1]
        y_probas = [
            torch.tensor([0.9, 0.1]),
            torch.tensor([0.2, 0.8]),
            torch.tensor([0.7, 0.3]),
            torch.tensor([0.4, 0.6]),
        ]
        indices = ["a", "b", "c", "d"]

        # Call the function
        report_and_eval.save_predictions_and_probs(save_dir, who, y_true, y_preds, y_probas, indices)

        # Check the results
        results_path = os.path.join(save_dir, f"{who}_predictions_and_probs.csv")
        self.assertTrue(os.path.exists(results_path), "Results file was not created.")

        # Read and verify the results
        results_df = pd.read_csv(results_path)
        # print(results_df)

        # Clean up
        os.remove(results_path)
        os.rmdir(save_dir)

        self.assertEqual(len(results_df), len(y_true), "Mismatch in number of rows saved.")
        self.assertListEqual(results_df["true"].tolist(), y_true, "True labels mismatch.")
        self.assertListEqual(results_df["predicted"].tolist(), y_preds, "Predicted labels mismatch.")
        self.assertListEqual(results_df["index"].tolist(), indices, "Indices mismatch.")
