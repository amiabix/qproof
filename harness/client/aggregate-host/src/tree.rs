//! Same split as pci_client::accumulator::fold. Positions are in roster order.
pub enum Step {
    Leaf(usize),
    Join,
}

pub fn plan(start: usize, count: usize, steps: &mut Vec<Step>) {
    assert!(count > 0);
    if count == 1 {
        steps.push(Step::Leaf(start));
    } else {
        let left = count.next_power_of_two() / 2;
        plan(start, left, steps);
        plan(start + left, count - left, steps);
        steps.push(Step::Join);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_rosters_have_the_expected_top_level_children() {
        for (n, split) in [(2usize, 1), (3, 2), (112, 64), (334, 256)] {
            let mut steps = Vec::new();
            plan(0, n, &mut steps);
            let mut ranges = Vec::new();
            let mut leaves = 0;
            assert_eq!(steps.len(), 2 * n - 1);
            for step in steps {
                match step {
                    Step::Leaf(i) => {
                        assert_eq!(i, leaves);
                        leaves += 1;
                        ranges.push((i, i + 1));
                    }
                    Step::Join => {
                        let (middle, end) = ranges.pop().unwrap();
                        let (start, left_end) = ranges.pop().unwrap();
                        assert_eq!(left_end, middle);
                        if (start, end) == (0, n) {
                            assert_eq!(middle, split);
                        }
                        ranges.push((start, end));
                    }
                }
            }
            assert_eq!(ranges, [(0, n)]);
        }
    }
}
