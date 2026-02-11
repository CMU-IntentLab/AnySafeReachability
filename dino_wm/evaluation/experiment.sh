for ex in 10 50 100 None; do
    for mrg in 0.5; do
        if [ "$ex" = "None" ]; then
            python train_failure_classifier_pa.py --dont-save-model --mrg "$mrg"
        else
            python train_failure_classifier_pa.py --dont-save-model --num-examples-per-class "$ex" --mrg "$mrg"
        fi
    done
done