#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>

/* Pinned c42 main listens with backlog 128 and exposes no CLI override.
 * Only this router process loads
 * the adapter; HTTP, request admission, inference and host sysctls are unchanged.
 * Linux still caps the requested backlog at net.core.somaxconn. */
static pthread_once_t once = PTHREAD_ONCE_INIT;
static int (*next_listen)(int, int);
static int minimum_backlog;

static void initialize(void) {
    next_listen = (int (*)(int, int))dlsym(RTLD_NEXT, "listen");
    const char *setting = getenv("AI_INFRA_LISTEN_BACKLOG");
    char *end = NULL;
    if (setting) {
        long value = strtol(setting, &end, 10);
        if (end != setting && *end == '\0' && value >= 128 && value <= 65535)
            minimum_backlog = (int)value;
    }
}

int listen(int socket_fd, int backlog) {
    pthread_once(&once, initialize);
    if (!next_listen) {
        errno = ENOSYS;
        return -1;
    }
    struct sockaddr_storage address;
    socklen_t length = sizeof(address);
    if (minimum_backlog > backlog &&
        getsockname(socket_fd, (struct sockaddr *)&address, &length) == 0 &&
        (address.ss_family == AF_INET || address.ss_family == AF_INET6)) {
        fprintf(stderr, "ai-infra router TCP listen backlog: %d -> %d\n",
                backlog, minimum_backlog);
        backlog = minimum_backlog;
    }
    return next_listen(socket_fd, backlog);
}
