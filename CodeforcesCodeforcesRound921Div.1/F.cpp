// time-limit: 2000
// problem-url: https://codeforces.com/contest/1924/problem/F
// 

#include <algorithm>
#include <array>
#include <bitset>
#include <cassert>
#include <chrono>
#include <climits>
#include <cmath>
#include <complex>
#include <cstring>
#include <cstdint>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <vector>
using namespace std;

using ll = long long;
using db = long double;
using str = string;

// pairs
using pi = pair<int, int>;
using pl = pair<ll, ll>;
using pd = pair<db, db>;
#define mp make_pair
#define f first
#define s second

#define tcT template <class T
#define tcTU tcT, class U

tcT > using V = vector<T>;
tcT, size_t SZ > using AR = array<T, SZ>;
using vi = V<int>;
using vb = V<bool>;
using vl = V<ll>;
using vd = V<db>;
using vs = V<str>;
using vpi = V<pi>;
using vpl = V<pl>;
using vpd = V<pd>;

// vectors
#define sz(x) int((x).size())
#define bg(x) begin(x)
#define all(x) bg(x), end(x)
#define rall(x) x.rbegin(), x.rend()
#define sor(x) sort(all(x))
#define rsz resize
#define ins insert
#define pb push_back
#define eb emplace_back
#define ft front()
#define bk back()

#define lb lower_bound
#define ub upper_bound
tcT > int lwb(V<T> &a, const T &b) { return int(lb(all(a), b) - bg(a)); }
tcT > int upb(V<T> &a, const T &b) { return int(ub(all(a), b) - bg(a)); }

// loops
#define FOR(i, a, b) for (int i = (a); i < (b); ++i)
#define F0R(i, a) FOR(i, 0, a)
#define ROF(i, a, b) for (int i = (b)-1; i >= (a); --i)
#define R0F(i, a) ROF(i, 0, a)
#define rep(a) F0R(_, a)
#define each(a, x) for (auto &a : x)

const int MOD = 998244353;  // 1e9+7;
const int MX = (int)2e5 + 5;
const ll BIG = 1e18;  // not too close to LLONG_MAX
const db PI = acos((db)-1);
const int dx[4]{1, 0, -1, 0}, dy[4]{0, 1, 0, -1};  // for every grid problem!!
mt19937 rng((uint32_t)chrono::steady_clock::now().time_since_epoch().count());
template <class T> using pqg = priority_queue<T, vector<T>, greater<T>>;

// bitwise ops
constexpr int pct(int x) { return __builtin_popcount(x); }  // # of bits set
constexpr int bits(int x) {

    return x == 0 ? 0 : 31 - __builtin_clz(x);
}
constexpr int p2(int x) { return 1 << x; }
constexpr int msk2(int x) { return p2(x) - 1; }

ll cdiv(ll a, ll b) {
    return a / b + ((a ^ b) > 0 && a % b);
}  // divide a by b rounded up
ll fdiv(ll a, ll b) {
    return a / b - ((a ^ b) < 0 && a % b);
}  // divide a by b rounded down

tcT > bool ckmin(T &a, const T &b) {
    return b < a ? a = b, 1 : 0;
}  // set a = min(a,b)
tcT > bool ckmax(T &a, const T &b) {
    return a < b ? a = b, 1 : 0;
}  // set a = max(a,b)

tcTU > T fstTrue(T lo, T hi, U f) {
    ++hi;
    assert(lo <= hi);  // assuming f is increasing
    while (lo < hi) {  // find first index such that f is true
        T mid = lo + (hi - lo) / 2;
        f(mid) ? hi = mid : lo = mid + 1;
    }
    return lo;
}
tcTU > T lstTrue(T lo, T hi, U f) {
    --lo;
    T d = 1;
    while (lo + d < hi) {
        if (!f(lo + d)) {
            hi = lo + d - 1;
            break;
        }
        d *= 2;
    }
    assert(lo <= hi);  // assuming f is decreasing
    while (lo < hi) {  // find first index such that f is true
        T mid = lo + (hi - lo + 1) / 2;
        f(mid) ? lo = mid : hi = mid - 1;
    }
    return lo;
}
tcT > void remDup(vector<T> &v) {  // sort and remove duplicates
    sort(all(v));
    v.erase(unique(all(v)), end(v));
}
tcTU > void safeErase(T &t, const U &u) {
    auto it = t.find(u);
    assert(it != end(t));
    t.erase(it);
}

inline namespace IO {
#define SFINAE(x, ...)                                                         \
    template <class, class = void> struct x : std::false_type {};              \
    template <class T> struct x<T, std::void_t<__VA_ARGS__>> : std::true_type {}

SFINAE(DefaultI, decltype(std::cin >> std::declval<T &>()));
SFINAE(DefaultO, decltype(std::cout << std::declval<T &>()));
SFINAE(IsTuple, typename std::tuple_size<T>::type);
SFINAE(Iterable, decltype(std::begin(std::declval<T>())));

template <auto &is> struct Reader {
    template <class T> void Impl(T &t) {
        if constexpr (DefaultI<T>::value) is >> t;
        else if constexpr (Iterable<T>::value) {
            for (auto &x : t) Impl(x);
        } else if constexpr (IsTuple<T>::value) {
            std::apply([this](auto &...args) { (Impl(args), ...); }, t);
        } else static_assert(IsTuple<T>::value, "No matching type for read");
    }
    template <class... Ts> void read(Ts &...ts) { ((Impl(ts)), ...); }
};

template <class... Ts> void re(Ts &...ts) { Reader<cin>{}.read(ts...); }
#define def(t, args...)                                                        \
    t args;                                                                    \
    re(args);

template <auto &os, bool debug, bool print_nd> struct Writer {
    string comma() const { return debug ? "," : ""; }
    template <class T> constexpr char Space(const T &) const {
        return print_nd && (Iterable<T>::value or IsTuple<T>::value) ? '\n'
                                                                     : ' ';
    }
    template <class T> void Impl(T const &t) const {
        if constexpr (DefaultO<T>::value) os << t;
        else if constexpr (Iterable<T>::value) {
            if (debug) os << '{';
            int i = 0;
            for (auto &&x : t)
                ((i++) ? (os << comma() << Space(x), Impl(x)) : Impl(x));
            if (debug) os << '}';
        } else if constexpr (IsTuple<T>::value) {
            if (debug) os << '(';
            std::apply(
                [this](auto const &...args) {
                    int i = 0;
                    (((i++) ? (os << comma() << " ", Impl(args)) : Impl(args)),
                     ...);
                },
                t);
            if (debug) os << ')';
        } else static_assert(IsTuple<T>::value, "No matching type for print");
    }
    template <class T> void ImplWrapper(T const &t) const {
        if (debug) os << "\033[0;31m";
        Impl(t);
        if (debug) os << "\033[0m";
    }
    template <class... Ts> void print(Ts const &...ts) const {
        ((Impl(ts)), ...);
    }
    template <class F, class... Ts>
    void print_with_sep(const std::string &sep, F const &f,
                        Ts const &...ts) const {
        ImplWrapper(f), ((os << sep, ImplWrapper(ts)), ...), os << '\n';
    }
    void print_with_sep(const std::string &) const { os << '\n'; }
};

template <class... Ts> void pr(Ts const &...ts) {
    Writer<cout, false, true>{}.print(ts...);
}
template <class... Ts> void ps(Ts const &...ts) {
    Writer<cout, false, true>{}.print_with_sep(" ", ts...);
}
}  // namespace IO

inline namespace Debug {
template <typename... Args> void err(Args... args) {
    Writer<cerr, true, false>{}.print_with_sep(" | ", args...);
}
template <typename... Args> void errn(Args... args) {
    Writer<cerr, true, true>{}.print_with_sep(" | ", args...);
}

void err_prefix(str func, int line, string args) {
    cerr << "\033[0;31m\u001b[1mDEBUG\033[0m"
         << " | "
         << "\u001b[34m" << func << "\033[0m"
         << ":"
         << "\u001b[34m" << line << "\033[0m"
         << " - "
         << "[" << args << "] = ";
}

#ifdef LOCAL
#define dbg(args...) err_prefix(__FUNCTION__, __LINE__, #args), err(args)
#define dbgn(args...) err_prefix(__FUNCTION__, __LINE__, #args), errn(args)
#else
#define dbg(...)
#define dbgn(args...)
#endif

const auto beg_time = std::chrono::high_resolution_clock::now();

double time_elapsed() {
    return chrono::duration<double>(std::chrono::high_resolution_clock::now() -
                                    beg_time)
        .count();
}
}  // namespace Debug

inline namespace FileIO {
void setIn(str s) { freopen(s.c_str(), "r", stdin); }
void setOut(str s) { freopen(s.c_str(), "w", stdout); }
void setIO(str s = "") {
    cin.tie(0)->sync_with_stdio(0);  // unsync C / C++ I/O streams
    cout << fixed << setprecision(12);

    if (sz(s)) setIn(s + ".in"), setOut(s + ".out");
}
}  // namespace FileIO

int calc_lim(int N) { return ceil(log(N) / log(1.116)) - 1; }
 
void solve(int tc) {
    def(int, N);
    vi hist(N);
    vi max_with_ans{2, 2, 2, 2};
    while (max_with_ans.bk < max(N, (int)1e5)) {
        int n = sz(max_with_ans);
        max_with_ans.pb((max_with_ans.at(n - 2) + max_with_ans.at(n - 3) +a
                         max_with_ans.at(n - 4)) /
                        2);
    }
    auto get_ans = [&](int len) {
        F0R(i, sz(max_with_ans)) if (len <= max_with_ans[i]) return i;
        assert(false);
    };
    int lim = get_ans(N);
    vi cands(N);
    iota(all(cands), 0);
    auto query = [&](int l, int r) -> bool {
        assert(lim);
        --lim;
        ps("?", 1 + l, 1 + r);
        cout.flush();
        int x;
        re(x);
        x = r - l + 1 - x;
        assert(0 <= x && x <= 1);
        vi ncands;
        for (int c : cands) {
            bool lie = x ^ (l <= c && c <= r);
            if (lie) {
                ckmin(hist[c], 0);
                --hist[c];
            } else {
                ckmax(hist[c], 0);
                ++hist[c];
            }
            if (abs(hist[c]) < 3) ncands.pb(c);
        }
        swap(cands, ncands);
        dbg(cands, hist, x);
        return x;
    };
    auto sgn = [&](int x) {
        if (x < 0) return -1;
        return 1;
    };
    auto all_same_sign = [&]() {
        assert(sz(cands));
        for (int c : cands) {
            if (hist[c] == 0) return false;
            if (sgn(hist[c]) != sgn(hist[cands.ft])) return false;
        }
        return true;
    };
    while (sz(cands) > 2) {
        const int len = sz(cands);
        const int a = get_ans(len);
        AR<vi, 3> segs;
        for (int c : cands) {
            if (len - sz(segs.at(0)) > max_with_ans.at(a - 2)) {
                segs.at(0).pb(c);
                continue;
            }
            if (len - sz(segs.at(1)) > max_with_ans.at(a - 3)) {
                segs.at(1).pb(c);
                continue;
            }
            segs.at(2).pb(c);
        }
        dbg(len, sz(segs[0]), sz(segs[1]), sz(segs[2]), a);
        assert(len - sz(segs.at(0)) <= max_with_ans.at(a - 2));
        assert(len - sz(segs.at(1)) <= max_with_ans.at(a - 3));
        assert(len - sz(segs.at(2)) <= max_with_ans.at(a - 4));
        if (!all_same_sign()) query(0, N - 1);
        assert(all_same_sign());
        assert(lim >= a - 1);
        vi state{sgn(hist[cands.ft])};
        while (sz(state) < 3) state.pb(state.bk);
        auto similar = [&](vi expected_state) {
            if (state != expected_state) { each(t, expected_state) t *= -1; }
            assert(state == expected_state);
        };
        auto upd = [&](int l, int r, bool res) {
            assert(0 <= l && l <= r && r <= 2);
            F0R(c, 3) {
                bool lie = (l <= c && c <= r) ^ res;
                if (lie) {
                    ckmin(state[c], 0);
                    --state[c];
                } else {
                    ckmax(state[c], 0);
                    ++state[c];
                }
            }
        };
        auto select = [&](int pos) {
            dbg("SELECT", pos);
            if (pos == -1) upd(0, 2, query(0, N - 1));
            else upd(pos, pos, query(segs.at(pos).ft, segs.at(pos).bk));
        };
        similar({1, 1, 1});
        select(0);
        dbg(state);
        if (abs(state[0]) == 2) {
            similar({-2, 1, 1});
            select(-1);
            if (abs(state[0]) == 3) {
                similar({-3, -1, -1});  
            } else {
                similar({1, 2, 2});
                select(1);
                if (abs(state[1]) == 3) {
                    similar({-1, 3, -1}); 
                } else {
                    similar({2, -1, 3}); 
                }
            }
        } else {
            similar({1, -2, -2});
            select(1);
            if (abs(state[1]) == 3) {
                similar({2, -3, 1}); 
            } else {
                similar({-1, 1, -3}); 
            }
        }
    }
    assert(sz(cands));
    for (int c : cands) {
        ps("!", 1 + c);
        cout.flush();
        def(int, y);
        if (y == 1) {
            ps("#");
            cout.flush();
            return;
        }
    }
    assert(false);
}

int main() {
    setIO();
    int TC;
    re(TC);
    FOR(i, 1, TC + 1) solve(i);
}














