// time-limit: 3000
// problem-url: https://codeforces.com/contest/1955/problem/H
//

#include <bits/stdc++.h>

using i64 = long long;

int srt[10000];

void solve() {
    int n, m, k;
    std::cin >> n >> m >> k;
    
    std::vector<std::string> s(n);
    for (int i = 0; i < n; i++) {
        std::cin >> s[i];
    }
    
    std::vector<int> dp(1 << 12);
    for (int t = 0; t < k; t++) {
        int x, y, p;
        std::cin >> x >> y >> p;
        x--, y--;
        
        std::vector<int> f(13);
        for (int a = 0; a < n; a++) {
            for (int b = 0; b < m; b++) {
                if (s[a][b] == '#') {
                    int r = srt[(a - x) * (a - x) + (b - y) * (b - y)];
                    if (r <= 12) {
                        f[r] += p;
                    }
                }
            }
        }
        for (int i = 1; i < f.size(); i++) {
            f[i] += f[i - 1];
        }
        int pw = 1;
        auto ndp = dp;
        for (int i = 0; i < f.size() && pw <= f.back(); i++) {
            if (f[i] - pw > 0) {
                assert(1 <= i && i <= 12);
                for (int s = 0; s < (1 << 12); s++) {
                    if (s >> (i - 1) & 1) {
                        ndp[s] = std::max(ndp[s], dp[s ^ (1 << (i - 1))] + f[i] - pw);
                    }
                }
            }
            pw *= 3;
        }
        dp = std::move(ndp);
    }
    
    std::cout << dp.back() << "\n";
}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);
    
    for (int x = 0, y = 0; x < 10000; x++) {
        if (y * y < x) {
            y++;
        }
        srt[x] = y;
        assert(int(std::ceil(std::sqrt(x))) == y);
    }
    
    int t;
    std::cin >> t;
    
    while (t--) {
        solve();
    }
    
    return 0;
}

