// https://atcoder.jp/contests/abc001/submissions/63920325
fn main() {
    let mut arr: Vec<u64> = vec![0; 1024 * 1024];
    for i in 1..128 {
        let mut page = vec![i as u64; 1024 * 1024];
        arr.extend(page);
    }
}
