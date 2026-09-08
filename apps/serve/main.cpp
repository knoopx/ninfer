#include "product/logging/logging.h"
#include "product/logging/startup_log.h"
#include "serve/generation_service.h"
#include "serve/http_server.h"
#include "serve/serve_options.h"

#include <spdlog/logger.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <limits.h>
#include <unistd.h>

namespace {

namespace fs = std::filesystem;

std::atomic<ninfer::serve::HttpServer*> g_server{nullptr};

void handle_signal(int) {
    ninfer::serve::HttpServer* server = g_server.load();
    if (server != nullptr) { server->stop(); }
}

// Self-locates the running executable and returns the directory containing it.
std::string executable_directory() {
    char buffer[PATH_MAX] = {};
    const ssize_t len = ::readlink("/proc/self/exe", buffer, sizeof(buffer) - 1);
    if (len <= 0) {
        throw std::runtime_error("readlink(/proc/self/exe) failed: cannot locate executable");
    }
    buffer[len] = '\0';
    return fs::path(buffer).parent_path().string();
}

// Returns the bundled webui directory: <executable-directory>/../share/ninfer/webui.
// The build copies the webui tree into that location at build time; the server only
// self-locates and verifies the static bundle (no version checking, no download).
std::string resolve_bundled_webui_dir() {
    const fs::path exe_dir = fs::path(executable_directory());
    return fs::weakly_canonical(exe_dir / ".." / "share" / "ninfer" / "webui").string();
}

// Verifies the bundled webui directory exists and contains a non-zero index.html.
void verify_bundled_webui(const std::string& dir) {
    const fs::path webui(dir);
    std::error_code ec;
    if (!fs::is_directory(webui, ec)) {
        throw std::runtime_error(
            "webui not bundled at " + dir +
            "; rebuild with a bundled webui (nix build .#ninfer).");
    }
    const fs::path index = webui / "index.html";
    std::error_code ec2;
    const auto size      = fs::file_size(index, ec2);
    if (ec2 || size == 0) {
        throw std::runtime_error(
            "webui bundled at " + dir +
            " has no valid index.html; rebuild with a bundled webui (nix build .#ninfer).");
    }
}

} // namespace

int main(int argc, char** argv) {
    ninfer::serve::ServeOptions options;
    try {
        options = ninfer::serve::parse_serve_options(argc, argv);
    } catch (const std::invalid_argument& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        std::cerr << ninfer::serve::serve_usage_text(argv[0]);
        return 1;
    } catch (const std::exception& exception) {
        std::cerr << "ninfer-serve: " << exception.what() << '\n';
        return 1;
    }
    if (options.help_requested) {
        std::cout << ninfer::serve::serve_usage_text(argv[0]);
        return 0;
    }

    ninfer::product::LoggingRuntime logging(
        {.logger_name  = "ninfer-serve",
         .level        = options.log_level,
         .presentation = ninfer::product::LogPresentation::Service});
    const std::shared_ptr<spdlog::logger> logger = logging.logger();
    ninfer::product::StartupLogRenderer startup_log(logging);
    ninfer::serve::OperationalLog operational_log(logger);
    bool serving = false;

    try {
        // When --webui is passed, resolve and verify the bundled webui directory
        // before the server is bound so a missing bundle aborts startup cleanly.
        if (options.webui_auto) {
            options.webui_dir = resolve_bundled_webui_dir();
            verify_bundled_webui(options.webui_dir);
        }
        // The router is held in a unique_ptr declared BEFORE the server so it is destroyed AFTER
        // the server (reverse declaration order). The server's destructor joins the httplib worker
        // threads, which releases any in-flight streaming Grants while the router is still alive
        // (a Grant's release_in_flight captures the router by raw pointer). The server binds
        // before the router is built; the router itself loads NO model at startup (no-resident).
        std::unique_ptr<ninfer::serve::ModelRouter> router;
        ninfer::serve::HttpServer server(options, logger);
        if (!server.bind()) {
            operational_log.bind_failure(options.host, options.port);
            return 1;
        }

        // In-process multi-model router (Option-B, NO-RESIDENT / on-demand-load): each model is a
        // GenerationService (Engine) built from a per-model ServeOptions (artifact_path pointed at
        // that model's artifact). The router launches UNLOADED -- nothing is pre-loaded or warmed
        // up at startup. The first request for a model loads it on demand (route() runs the load
        // path from the empty state + the /health readiness gate); a loaded model persists until
        // its TTL evicts it, then the next request reloads it. server.bind() already happened
        // above, so the router is bind()-independent.
        auto make_backend =
            [&options, &startup_log, &operational_log](const ninfer::serve::ModelConfig& config)
                -> std::unique_ptr<ninfer::serve::ModelBackend> {
                auto backend = std::make_unique<ninfer::serve::EngineModelBackend>(
                    config, options, startup_log.observer());
                // Log the complete preset parameter set of the loaded model (identity, the
                // normalized engine parameters, the shared memory/ingress values, and which
                // per-model overrides the serve config set) so every on-demand load / swap-in
                // records the exact configuration the Engine was constructed with.
                operational_log.model_preset(*backend, config, options);
                return backend;
            };
        router =
            std::make_unique<ninfer::serve::ModelRouter>(options, startup_log.observer(),
                                                         std::move(make_backend));
        // No-resident, on-demand-load: the server launches UNLOADED. The router holds no loaded
        // backend yet (router->resident() is null at this point -- that is the expected state,
        // not an error). Nothing is warmed up; the first request for a model loads it on demand.
        startup_log.engine_ready(ninfer::LoadSummary{}); // "no model loaded yet" (on-demand)
        server.attach(*router);

        g_server.store(&server);
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);

        serving = true;
        operational_log.server_ready(options.host, options.port, router->loaded_id(),
                                     !options.api_key.empty());

        const bool ok = server.listen();
        g_server.store(nullptr);
        if (!ok) {
            operational_log.listen_failure(options.host, options.port);
            return 1;
        }
        operational_log.server_stopped();
        return 0;
    } catch (const std::exception& exception) {
        g_server.store(nullptr);
        operational_log.server_failure(serving, exception.what());
        return 1;
    }
}
