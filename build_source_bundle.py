"""Build a reviewable allowlisted source archive; never include speech or weights."""
from pathlib import Path
import hashlib,json,zipfile

ROOT=Path(__file__).resolve().parent
def main():
    paths=[ROOT/'README.md',ROOT/'ROADMAP.md',ROOT/'MIGRATION.md',ROOT/'Open_VocalMap.ipynb',ROOT/'.gitignore',ROOT/'.github/workflows/validate.yml',Path(__file__).resolve()]
    for module in ('vm_acoustic','vm_perceptual'):
        for pattern in ('*.py','*.md','requirements.txt','.gitignore'):
            paths.extend(p for p in (ROOT/module).glob(pattern) if p.name != 'local_language_audit.py')
    for name in ('REPORT.md','evaluation.json','rating_geometry.json','short_clip_check.json','adapter_check.json','resume_check.json','dev_text_check.json','dev_whitening.json',
                 'raw5090_comparison.json','fixed5090_comparison.json','precision_error_breakdown.json',
                 'fresh_text_development.json','cross_text_development.json','diagonal_metric_dev.json',
                 'content_conditioning.json','layer_metric_dev.json','frozen_new_texts.json','vctk_map_stability.json',
                 'blend_projection.json','crop_stability.json','paired_correction.json','trimmed_input_stability.json',
                 'sliced_spectra.json','nested_people_summary.json','crop_estimator.json','input_stability.json','pitch_audit.json',
                 'wave_comparison.json','wave_style_audit.json','reference_benchmark.json','source_filter_dev.json',
                 'map_retrieval_eval.json','recording_transfer.json','gallery_benchmark.json','window_aggregation_dev.json',
                 'window_aggregation_test.json','redim_reference.json','rank_map_candidate.json','display_geometry.json',
                 'rank_confirmation.json','repeated_fold_studies.json','distilled_confirmation.json','rating_repeatability.json',
                 'crop_uncertainty.json','morph_paths.json','rank_cross_hardware.json','map_consensus.json',
                 'final_study_summary.json','robust_ssl_summary.json'):
        p=ROOT/'vm_perceptual/results'/name
        if not p.is_file():
            p=ROOT/'vm_perceptual/reference_results'/name
        if p.is_file():paths.append(p)
    for folder in ('results_reference','results_confirmation','results_sentence_validation'):
        base=ROOT/'vm_acoustic'/folder
        # Selected summaries only; no caches, per-frame observations or audio.
        for name in ('report.md','metrics.json','tracker_experiment.json','tracker_child_challenge.json',
                     'tracker_metrics.json','coverage_audit.json','equalization_report.md','locked_method.json'):
            if (base/name).is_file():paths.append(base/name)
    out=ROOT/'deliverables';out.mkdir(exist_ok=True)
    target=out/'vocalmap-research-source.zip'
    manifest={}
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(set(paths)):
            relative=p.relative_to(ROOT).as_posix()
            # Historical scores must not occupy the fresh run's guarded output path.
            if relative.startswith('vm_perceptual/results/'):
                relative=relative.replace('/results/', '/reference_results/', 1)
            data=p.read_bytes().replace(b'\r\n',b'\n')  # Match Git text blobs on Windows and Linux.
            if p.suffix=='.ipynb':
                notebook=json.loads(data)
                if any(cell.get('outputs') for cell in notebook['cells']):
                    raise ValueError('Clear notebook outputs before creating a public source bundle')
            manifest[relative]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            z.writestr(relative,data)
        z.writestr('SOURCE_MANIFEST.json',json.dumps(manifest,indent=2,ensure_ascii=False))
    with zipfile.ZipFile(target) as z:
        assert z.testzip() is None
        assert all(Path(n).suffix.lower() not in ('.wav','.flac','.pt','.dll') for n in z.namelist())
        assert not any(n.startswith(('ESD/','claude_baselines/')) or '/data_jvs/' in n or '/data_sources/' in n for n in z.namelist())
        for name,m in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==m['sha256']
    digest=hashlib.sha256(target.read_bytes()).hexdigest()
    (out/'source_bundle.json').write_text(json.dumps(dict(file=target.name,bytes=target.stat().st_size,
        sha256=digest,files=len(manifest),archive_integrity='passed',excludes='audio, dependencies, weights, private collaboration logs'),indent=2))
    print(target, len(manifest),'files',target.stat().st_size,'bytes',digest)

if __name__=='__main__':main()
